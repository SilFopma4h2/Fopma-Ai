"""
Enhanced text generation with multiple sampling strategies
"""

import torch
import torch.nn.functional as F


class TextGenerator:
    """Enhanced text generator with multiple sampling strategies"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def generate_text(self, prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.1):
        """Generate text using enhanced sampling strategies"""
        self.model.eval()
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Store original prompt length
        prompt_length = input_ids.shape[1]
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                logits = self.model(input_ids)
                next_token_logits = logits[0, -1, :]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits, input_ids, repetition_penalty
                    )
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    next_token_logits = self._top_k_filtering(next_token_logits, top_k)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    next_token_logits = self._top_p_filtering(next_token_logits, top_p)
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
                
                # Check for end of sequence
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode and return generated text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text
    
    def _apply_repetition_penalty(self, logits, input_ids, penalty):
        """Apply repetition penalty to discourage repetitive text"""
        for token_id in set(input_ids[0].tolist()):
            if logits[token_id] > 0:
                logits[token_id] = logits[token_id] / penalty
            else:
                logits[token_id] = logits[token_id] * penalty
        return logits
    
    def _top_k_filtering(self, logits, top_k):
        """Apply top-k filtering"""
        if top_k > 0:
            # Get top-k values and indices
            top_k = min(top_k, logits.size(-1))
            top_k_values, _ = torch.topk(logits, top_k)
            
            # Set all other values to negative infinity
            min_value = top_k_values[-1]
            logits = torch.where(logits < min_value, torch.full_like(logits, -float('inf')), logits)
        
        return logits
    
    def _top_p_filtering(self, logits, top_p):
        """Apply top-p (nucleus) filtering"""
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        
        # Create mask for original indices
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('inf')
        
        return logits
    
    def chat_interface(self):
        """Interactive chat interface"""
        print("🤖 Enhanced Mini-ChatGPT Chat Interface")
        print("=" * 50)
        print("Commands:")
        print("  'quit' or 'exit' - Exit the chat")
        print("  'clear' - Clear conversation history")
        print("  'settings' - Adjust generation parameters")
        print("=" * 50)
        
        # Default generation settings
        settings = {
            'max_length': 100,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'repetition_penalty': 1.1
        }
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye! Thanks for chatting with Fopma-AI!")
                    break
                
                elif user_input.lower() == 'clear':
                    conversation_history = []
                    print("🗑️ Conversation history cleared!")
                    continue
                
                elif user_input.lower() == 'settings':
                    self._adjust_settings(settings)
                    continue
                
                # Add user input to conversation history
                conversation_history.append(f"Human: {user_input}")
                
                # Create context from recent conversation
                context = self._create_context(conversation_history, max_context_length=3)
                prompt = f"{context}\nAI:"
                
                # Generate response
                print("🤖 AI: ", end="", flush=True)
                
                response = self.generate_text(
                    prompt,
                    max_length=settings['max_length'],
                    temperature=settings['temperature'],
                    top_k=settings['top_k'],
                    top_p=settings['top_p'],
                    repetition_penalty=settings['repetition_penalty']
                )
                
                # Extract just the AI's response
                ai_response = response[len(prompt):].strip()
                
                # Clean up the response
                ai_response = self._clean_response(ai_response)
                
                print(ai_response)
                
                # Add AI response to conversation history
                conversation_history.append(f"AI: {ai_response}")
                
                # Keep conversation history manageable
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-8:]
                
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.")
    
    def _create_context(self, conversation_history, max_context_length=3):
        """Create context from conversation history"""
        if not conversation_history:
            return ""
        
        # Take the last few exchanges
        recent_history = conversation_history[-max_context_length*2:]
        return "\n".join(recent_history)
    
    def _clean_response(self, response):
        """Clean up the AI response"""
        # Remove any remaining prompt artifacts
        response = response.split("Human:")[0].split("AI:")[0]
        
        # Remove excessive whitespace
        response = " ".join(response.split())
        
        # Ensure reasonable length
        if len(response) > 500:
            # Find the last complete sentence within reasonable length
            sentences = response.split('.')
            cleaned = ""
            for sentence in sentences:
                if len(cleaned + sentence + ".") <= 500:
                    cleaned += sentence + "."
                else:
                    break
            response = cleaned if cleaned else response[:500]
        
        return response.strip()
    
    def _adjust_settings(self, settings):
        """Interactive settings adjustment"""
        print("\n⚙️ Current Generation Settings:")
        for key, value in settings.items():
            print(f"   {key}: {value}")
        
        print("\nAdjust settings (press Enter to keep current value):")
        
        try:
            # Max length
            new_max_length = input(f"Max length ({settings['max_length']}): ").strip()
            if new_max_length:
                settings['max_length'] = max(10, min(500, int(new_max_length)))
            
            # Temperature
            new_temperature = input(f"Temperature ({settings['temperature']}): ").strip()
            if new_temperature:
                settings['temperature'] = max(0.1, min(2.0, float(new_temperature)))
            
            # Top-k
            new_top_k = input(f"Top-k ({settings['top_k']}): ").strip()
            if new_top_k:
                settings['top_k'] = max(1, min(100, int(new_top_k)))
            
            # Top-p
            new_top_p = input(f"Top-p ({settings['top_p']}): ").strip()
            if new_top_p:
                settings['top_p'] = max(0.1, min(1.0, float(new_top_p)))
            
            # Repetition penalty
            new_rep_penalty = input(f"Repetition penalty ({settings['repetition_penalty']}): ").strip()
            if new_rep_penalty:
                settings['repetition_penalty'] = max(1.0, min(2.0, float(new_rep_penalty)))
            
            print("✅ Settings updated!")
            
        except ValueError:
            print("❌ Invalid input. Settings unchanged.")
    
    def generate_examples(self, prompts=None):
        """Generate example outputs for given prompts"""
        if prompts is None:
            prompts = [
                "The future of artificial intelligence is",
                "In a world where machines can think",
                "The most important aspect of learning is",
                "Technology has changed our lives by"
            ]
        
        print("🎯 Generating example outputs...")
        print("=" * 50)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n📝 Example {i}: '{prompt}'")
            print("🤖 Generated text:")
            
            generated = self.generate_text(prompt, max_length=80, temperature=0.7)
            print(f"   {generated}")
            print("-" * 40)