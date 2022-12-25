





if __name__ == '__main__':
    bart_name = ''

    model = BartSeq2SeqModel.build_model(bart_name, tokenizer, label_ids=label_ids, 
                    decoder_type='avg_score', copy_gate=False, use_encoder_mlp=True, use_recur_pos=False)
    vocab_size = len(tokenizer)
    model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                                eos_token_id=eos_token_id,
                                max_length=max_len, max_len_a=max_len_a,num_beams=num_beams, do_sample=False,
                                repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_id,
                                restricter=None)
    
