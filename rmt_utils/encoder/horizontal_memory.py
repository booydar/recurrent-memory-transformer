import torch
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

def horizontal_memory_forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        rmt_parent = None
    ):
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

    next_decoder_cache = () if use_cache else None
    for i, layer_module in enumerate(self.layer):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[i] if head_mask is not None else None
        past_key_value = past_key_values[i] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            if use_cache:
                raise Warning("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, past_key_value, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            if i in rmt_parent.memory_storage:
                layer_memory = rmt_parent.memory_storage[i]
                non_empty_mask = rmt_parent.memory_storage['non_empty_mask']
                if layer_memory.shape[0] == 1:
                    layer_memory = layer_memory.repeat(len(non_empty_mask), 1, 1)
                    
                hidden_states = torch.cat([layer_memory[non_empty_mask], hidden_states], dim=1)
                if attention_mask is not None:
                    layer_attention_mask = torch.cat((attention_mask[:, :, :, :rmt_parent.num_mem_tokens], attention_mask), dim=-1)
                else:
                    layer_attention_mask = attention_mask
            else:
                layer_memory = None
                layer_attention_mask = attention_mask

            layer_outputs = layer_module(
                hidden_states,
                layer_attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            ### Update memory
            if rmt_parent.memory_layers is not None:
                memory_layer = rmt_parent.memory_layers[i]
                memory_layer_out = memory_layer(
                    hidden_states,
                    layer_attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

        hidden_states = layer_outputs[0]
        
        ### shorten hidden states
        if i in rmt_parent.memory_storage:
            hidden_states = hidden_states[:, rmt_parent.num_mem_tokens:]

        ### update memory 
        if rmt_parent.memory_layers is not None:
            memory_layer_hidden_states = memory_layer_out[0]
            if i in rmt_parent.memory_storage:
                memory_layer_hidden_states = memory_layer_hidden_states[:, rmt_parent.num_mem_tokens:]

            updated_memory = memory_layer_hidden_states[:, rmt_parent.memory_position]
            hidden_states[:, rmt_parent.memory_position] = updated_memory
        
        ### set layer memory
        if layer_memory is not None:
            layer_memory[non_empty_mask] = hidden_states[:, rmt_parent.memory_position].detach()
        else:
            layer_memory = hidden_states[:, rmt_parent.memory_position].detach()
        rmt_parent.memory_storage[i] = layer_memory

        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )


from collections.abc import Sequence
from transformers.modeling_outputs import BaseModelOutput

def deberta_horizontal_memory_forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
        rmt_parent=None,
    ):
    attention_mask = self.get_attention_mask(attention_mask)
    relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    if isinstance(hidden_states, Sequence):
        next_kv = hidden_states[0]
    else:
        next_kv = hidden_states
    rel_embeddings = self.get_rel_embedding()
    for i, layer_module in enumerate(self.layer):

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if i in rmt_parent.memory_storage:
            layer_memory = rmt_parent.memory_storage[i]
            non_empty_mask = rmt_parent.memory_storage['non_empty_mask']
            if layer_memory.shape[0] == 1:
                layer_memory = layer_memory.repeat(len(non_empty_mask), 1, 1)
                
            hidden_states = torch.cat([layer_memory[non_empty_mask], hidden_states], dim=1)
            if attention_mask is not None:
                layer_attention_mask = torch.cat((attention_mask, attention_mask[:, :, :, :rmt_parent.num_mem_tokens]), dim=-1)
            else:
                layer_attention_mask = attention_mask
        else:
            layer_memory = None
            layer_attention_mask = attention_mask

        hidden_states = layer_module(
            next_kv,
            attention_mask=layer_attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        
        ### Update memory
        if rmt_parent.memory_layers is not None:
            memory_layer = rmt_parent.memory_layers[i]
            memory_layer_out = memory_layer(
                next_kv,
                attention_mask=layer_attention_mask,
                output_attentions=output_attentions,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
            )

        if output_attentions:
            hidden_states, att_m = hidden_states
            memory_layer_out, memory_attentions = memory_layer_out
        
        ### shorten hidden states
        if i in rmt_parent.memory_storage:
            hidden_states = hidden_states[:, rmt_parent.num_mem_tokens:]

        ### update memory 
        if rmt_parent.memory_layers is not None:
            memory_layer_hidden_states = memory_layer_out[0]
            if i in rmt_parent.memory_storage:
                memory_layer_hidden_states = memory_layer_hidden_states[:, rmt_parent.num_mem_tokens:]

            updated_memory = memory_layer_hidden_states[:, rmt_parent.memory_position]
            hidden_states[:, rmt_parent.memory_position] = updated_memory
        
        ### set layer memory
        if layer_memory is not None:
            layer_memory[non_empty_mask] = hidden_states[:, rmt_parent.memory_position].detach()
        else:
            layer_memory = hidden_states[:, rmt_parent.memory_position].detach()
        rmt_parent.memory_storage[i] = layer_memory

        if query_states is not None:
            query_states = hidden_states
            if isinstance(hidden_states, Sequence):
                next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
        else:
            next_kv = hidden_states

        if output_attentions:
            all_attentions = all_attentions + (att_m,)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
    return BaseModelOutput(
        last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
    )