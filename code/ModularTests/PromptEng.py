class PromptEng:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def combine_prompt(self, external_context, user_prompt):
        combined_prompt = f"""
        Context: {external_context}

        User Question: {user_prompt}

        Please provide a detailed and natural-sounding answer based on the context above. 
        Maintain all medical terminology and ensure the response is clear and concise.
        
        Answer:
        """
        return combined_prompt

    def generate_output(self, combined_prompt):
        inputs = self.tokenizer(combined_prompt, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.35,
            top_p=0.85
        )
        generated_tokens = outputs[0]
        return generated_tokens

    def process(self, external_context, user_prompt):
        combined_prompt = self.combine_prompt(external_context, user_prompt)
        generated_tokens = self.generate_output(combined_prompt)
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        answer = generated_text.split("Answer:")[1].strip() if "Answer:" in generated_text else generated_text
        return answer

