import math

import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from transformers.activations import ACT2FN


class MemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens=None, num_read_mem_tokens=None, num_write_mem_tokens=None):
        super().__init__()
        self.model = base_model
        self.num_mem_tokens = num_mem_tokens
        self.num_read_mem_tokens = num_read_mem_tokens
        self.num_write_mem_tokens = num_write_mem_tokens

        if self.num_read_mem_tokens is None:
            self.num_read_mem_tokens = num_mem_tokens

        if self.num_write_mem_tokens is None:
            self.num_write_mem_tokens = num_mem_tokens

        # let's define number of memory tokens as number of memory tokens that model outputs (writes to)
        self.num_mem_tokens = self.num_write_mem_tokens

        self.create_memory()

    def create_memory(self):
        embeddings = self.model.get_input_embeddings()
        memory_dim = getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
        memory_weights = torch.randn((self.num_read_mem_tokens, memory_dim)) * embeddings.weight.data.std()
        self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))

        self.read_memory_position = range(self.num_read_mem_tokens)
        self.write_memory_position = range(-self.num_write_mem_tokens, 0)

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def forward(self, input_ids, memory_state=None, **kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, write_mem=True, **kwargs)
        out = self.model(**seg_kwargs)
        out, new_memory_state = self.process_output(out, **kwargs)

        return out, new_memory_state

    def generate(self, input_ids, memory_state, attention_mask, **generate_kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, attention_mask=attention_mask, write_mem=False)
        out = self.model.generate(inputs_embeds=seg_kwargs['inputs_embeds'], attention_mask=seg_kwargs['attention_mask'], **generate_kwargs)
        return out

    def process_input(self, input_ids, memory_state, write_mem, **kwargs):
        seg_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if self.num_mem_tokens > 0:
            # memory_state: [recurrent_memory; retrieved_memory]
            # write memory == recurrent memory, it updates recurrent state, read memory can be:
            #   - [recurrent_memory; retrieved_memory]
            #   or
            #   - [retrieved_memory]
            read_memory_state = memory_state[:, -self.num_read_mem_tokens:, :]
            write_memory_state = memory_state[:, :self.num_write_mem_tokens, :]
            if write_mem:
                inputs_embeds = torch.cat([read_memory_state, inputs_embeds, write_memory_state], dim=1)
            else:
                inputs_embeds = torch.cat([read_memory_state, inputs_embeds], dim=1)

        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        if kwargs.get('attention_mask') is not None:
            seg_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape)
        seg_kwargs['output_hidden_states'] = True
        return seg_kwargs

    def pad_attention_mask(self, attention_mask, shape):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            mask[:, self.num_read_mem_tokens: self.num_read_mem_tokens + attention_mask.shape[1]] = attention_mask
            return mask

    def process_output(self, model_outputs, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            memory_state = model_outputs.hidden_states[-1][:, -self.num_write_mem_tokens:]
            out['logits'] = model_outputs.logits[:, self.num_read_mem_tokens:-self.num_write_mem_tokens]

            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = [lh[:, self.num_read_mem_tokens:-self.num_write_mem_tokens] for lh in model_outputs.hidden_states]
            if kwargs.get('output_attentions'):
                out['attentions'] = model_outputs['attentions']
        else:
            memory_state = None
            out = model_outputs

        return out, memory_state


class RecurrentWrapper(torch.nn.Module):
    def __init__(self, memory_cell, **rmt_kwargs):
        super().__init__()
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs
        self.memory_dim = self.memory_cell.memory.shape[-1]
        self.retrieve_mode = self.rmt_config['retrieve_mode']  # 'attention' # top_1
        self.intermediate_size = self.rmt_config.get('rmt_intermediate_size', self.memory_dim * 4)
        self.attention_dropout_prob = self.rmt_config.get('rmt_attention_dropout', 0.1)
        self.hid_dropout_prob = self.rmt_config.get('rmt_hidden_dropout', 0.1)
        self.layer_norm_eps = self.rmt_config.get('rmt_layer_norm_eps', 1e-5)

        # Q, K, V and output proj
        self.q_proj = torch.nn.Linear(self.memory_dim, self.memory_dim)
        self.k_proj = torch.nn.Linear(self.memory_dim, self.memory_dim)
        self.v_proj = torch.nn.Linear(self.memory_dim, self.memory_dim)
        self.o_proj = torch.nn.Linear(self.memory_dim, self.memory_dim)
        # MLP
        self.fc_1 = torch.nn.Linear(self.memory_dim, self.intermediate_size)
        self.fc_2 = torch.nn.Linear(self.intermediate_size, self.memory_dim)
        self.att_dropout = torch.nn.Dropout(self.attention_dropout_prob)
        self.dropout_1 = torch.nn.Dropout(self.hid_dropout_prob)
        self.dropout_2 = torch.nn.Dropout(self.hid_dropout_prob)
        self.ln_q = torch.nn.LayerNorm(self.memory_dim, eps=self.layer_norm_eps)
        self.ln_kv = torch.nn.LayerNorm(self.memory_dim, eps=self.layer_norm_eps)
        self.ln_mlp = torch.nn.LayerNorm(self.memory_dim, eps=self.layer_norm_eps)
        self.act = ACT2FN[getattr(self.memory_cell.model.config, 'activation_function', 'gelu')]

    def retrieve_from_past_memory_states(self, past_states, current_state):
        if len(past_states) == 0:
            return None

        # smth like single-head cross-attention between current state and past states
        # get query from current states
        q = self.q_proj(self.ln_q(current_state))
        hiddens = torch.cat(past_states, dim=1)
        hiddens = self.ln_kv(hiddens)
        # get keys and values from past states
        k = self.k_proj(hiddens)
        v = self.v_proj(hiddens)
        att_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.memory_dim)

        if self.retrieve_mode == 'attention':
            att_probs = torch.nn.functional.softmax(att_scores, dim=-1)
            att_probs = self.att_dropout(att_probs)
        elif self.retrieve_mode == 'top_1':
            # top_1_indices = torch.argmax(att_probs, dim=-1)
            # top_1_ohe = torch.nn.functional.one_hot(top_1_indices, num_classes=att_scores.shape[2]).type(v.dtype)
            # retrieved_values = torch.matmul(top_1_ohe, v)
            # use softmax with very low temperature to make it looks like [0, 0, 1, 0, ..., 0]
            att_probs = torch.nn.functional.softmax(att_scores * 1e05, dim=-1)

        retrieved_values = torch.matmul(att_probs, v)
        retrieved_values = self.o_proj(retrieved_values)
        retrieved_values = self.dropout_1(retrieved_values)
        # skip connection over cross-attention
        retrieved_values = retrieved_values + current_state
        # mem_dim -> interm_dim -> mem_dim
        mlp_hiddens = self.fc_2(self.act(self.fc_1(self.ln_mlp(retrieved_values))))
        mlp_hiddens = self.dropout_2(mlp_hiddens)
        # skip connection over MLP (?)
        retrieved_values = retrieved_values + mlp_hiddens
        return retrieved_values

    def forward(self, input_ids, labels=None, labels_mask=None, inputs_embeds=None, attention_mask=None, output_attentions=None, output_hidden_states=None):
        memory_state = None
        memory_states = []
        segmented = self.segment(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        cell_outputs = []
        for seg_num, segment in enumerate(segmented):
            retrieved_memory = self.retrieve_from_past_memory_states(memory_states, memory_state)
            if memory_state is not None and retrieved_memory is not None:
                memory_state = torch.cat([memory_state, retrieved_memory], dim=1)
            cell_out, memory_state = self.memory_cell(**segment, memory_state=memory_state, output_hidden_states=True)
            cell_outputs.append(cell_out)
            memory_state = self.manage_gradients(memory_state, seg_num)
            if self.rmt_config.get('k2', -1) == -1:
                memory_states += [memory_state]
            else:
                raise NotImplementedError  # check how to detach here?

        out = self.process_outputs(cell_outputs, labels=labels,
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states)
        return out

    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        memory_state = None
        memory_states = []
        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)

        for seg_num, segment in enumerate(segmented[:-1]):
            retrieved_memory = self.retrieve_from_past_memory_states(memory_states, memory_state)
            if memory_state is not None and retrieved_memory is not None:
                memory_state = torch.cat([memory_state, retrieved_memory], dim=1)
            cell_out, memory_state = self.memory_cell(**segment, memory_state=memory_state, output_hidden_states=True)
            memory_states += [memory_state.detach()]

        final_segment = segmented[-1]

        retrieved_memory = self.retrieve_from_past_memory_states(memory_states, memory_state)
        if memory_state is not None and retrieved_memory is not None:
            memory_state = torch.cat([memory_state, retrieved_memory], dim=1)
        out = self.memory_cell.generate(**final_segment, memory_state=memory_state, **generate_kwargs)

        return out

    def segment(self, **kwargs):
        segments = []
        for k, tensor in kwargs.items():
            if tensor is not None:
                k_segments = self.split_tensor(tensor)
                for s, k_seg in enumerate(k_segments):
                    if s < len(segments):
                        segments[s][k] = k_seg
                    else:
                        segments.append({k: k_seg})

        return segments

    def split_tensor(self, tensor):
        align = self.rmt_config.get('segment_alignment', 'left')
        segment_size = self.rmt_config.get('segment_size')
        if align == 'left':
            split_inds = list(range(0, tensor.shape[1], segment_size)) + [tensor.shape[1]]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align == 'right':
            split_inds = (list(range(tensor.shape[1], 0, -segment_size)) + [0])[::-1]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align == 'center':
            n_seg = math.ceil(tensor.shape[1] / segment_size)
            segments = torch.chunk(tensor, n_seg, dim=1)
        else:
            raise NotImplementedError(f'segment_alignment=={align}')
        return segments

    def process_outputs(self, cell_outputs, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        full_hidden_states = tuple([torch.cat(layer_hs, dim=1) for layer_hs in zip(*[o.hidden_states for o in cell_outputs])])

        labels = kwargs.get('labels')
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = full_logits[..., :-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))

            loss_fct = CrossEntropyLoss()
            labels_mask = kwargs.get('labels_mask')
            if labels_mask is not None:
                shift_mask = labels_mask[..., :-1].contiguous()

                flat_labels = flat_labels[shift_mask.view(-1)]
                flat_logits = flat_logits[shift_mask.view(-1)]

            out['loss'] = loss_fct(flat_logits, flat_labels)
            if out['loss'] is None:
                raise ValueError
        else:
            out['loss'] = 0

        out['logits'] = full_logits
        segment_keys = ['loss', 'logits']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')
        if kwargs.get('output_hidden_states'):
            segment_keys.append('hidden_states')
            out['hidden_states'] = full_hidden_states

        for seg_num, o in enumerate(cell_outputs):
            for key, value in o.items():
                if any([sk in key for sk in segment_keys]):
                    out[f'{key}_{seg_num}'] = value

        return out

    def manage_gradients(self, memory_state, seg_num):
        k2, max_n_segments = self.rmt_config.get('k2'), self.rmt_config.get('max_n_segments')
        if seg_num == 0 \
            or k2 in {-1, None} \
            or seg_num + k2 > max_n_segments:
                return memory_state

        memory_state = memory_state.detach()
        return memory_state


class RecurrentWrapperLight(RecurrentWrapper):    
    def forward(self, input_ids, labels=None, labels_mask=None, inputs_embeds=None, attention_mask=None, output_attentions=None, output_hidden_states=None):
        memory_state = None
        memory_states = []
        segmented = self.segment(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        cell_outputs = []
        for seg_num, segment in enumerate(segmented):
            retrieved_memory = self.retrieve_from_past_memory_states(memory_states, memory_state)
            if memory_state is not None and retrieved_memory is not None:
                memory_state = torch.cat([memory_state, retrieved_memory], dim=1)
            cell_out, memory_state = self.memory_cell(**segment, memory_state=memory_state, output_hidden_states=True)
            memory_state = self.manage_gradients(memory_state, seg_num)
            if self.rmt_config.get('k2', -1) == -1:
                memory_states += [memory_state]
            else:
                raise NotImplementedError  # check how to detach here?
        cell_outputs = [cell_out]

        out = self.process_outputs(cell_outputs, labels=labels,
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states)
        return out

    def process_outputs(self, cell_outputs, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        full_hidden_states = tuple([torch.cat(layer_hs, dim=1) for layer_hs in zip(*[o.hidden_states for o in cell_outputs])])

        labels = kwargs.get('labels')
        if labels is not None:
            shift_labels = labels[..., 1:]
            shift_logits = full_logits[..., :-1, :].contiguous()
            shift_labels = shift_labels[:, -shift_logits.shape[1]:].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))

            loss_fct = CrossEntropyLoss()
            labels_mask = kwargs.get('labels_mask')
            if labels_mask is not None:
                shift_mask = labels_mask[..., :-1]
                shift_mask = shift_mask[:, -shift_logits.shape[1]:].contiguous()

                flat_labels = flat_labels[shift_mask.view(-1)]
                flat_logits = flat_logits[shift_mask.view(-1)]

            out['loss'] = loss_fct(flat_logits, flat_labels)
            if out['loss'] is None:
                raise ValueError
        else:
            out['loss'] = 0

        out['logits'] = full_logits
        segment_keys = ['loss', 'logits']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')
        if kwargs.get('output_hidden_states'):
            segment_keys.append('hidden_states')
            out['hidden_states'] = full_hidden_states

        for seg_num, o in enumerate(cell_outputs):
            for key, value in o.items():
                if any([sk in key for sk in segment_keys]):
                    out[f'{key}_{seg_num}'] = value

        return out
