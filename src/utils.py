from torch.utils.data import Dataset
import torch
import json
import re
import torch.nn.functional as F
from tokenizers import pre_tokenizers, Regex
from tqdm.auto import tqdm

INSTRUCT_PROMPT = "Generate the entire rest of this text, continuing until you reach the end: "
INSTRUCT_PROMPT_PREFIX = 15

class CustomDataset(Dataset):

    def __init__(self, dataset, tokenizer, model, unmemorize=False, 
                 unmemorize_start = 7, unmemorize_stride = 8, unmemorize_span = 1, 
                 unmemorize_smart_stride = False,
                 unmemorize_smart_select = False,
                 unmemorize_top_k = 10,
                 max_length=512, instruct=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model = model.to("cuda")
        self.max_length = max_length
        self.instruct = instruct
        
        # save the unmemorize parameters
        self.unmemorize_start = unmemorize_start
        self.unmemorize_stride = unmemorize_stride
        self.unmemorize_span = unmemorize_span
        self.unmemorize_smart_stride = unmemorize_smart_stride
        self.unmemorize_smart_select = unmemorize_smart_select
        self.unmemorize_top_k = unmemorize_top_k
        self.answer_index = []
        
        # tokenize the articles ommitting special tokens
        self.tokenized_articles = [tokenizer.encode(article, add_special_tokens=False) for article in dataset['article']]
        
        # Tokenize all articles at once
        if self.instruct:
            prompts = [' '.join(item['article'].split()[:INSTRUCT_PROMPT_PREFIX]) for item in self.dataset]
            conversations = []
            
            for prompt, item in zip(prompts, self.dataset):
                prompt_conv = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": f"{INSTRUCT_PROMPT} {prompt}"}
                ]
                prompt_only = self.tokenizer.apply_chat_template(prompt_conv, tokenize=False, add_generation_prompt=True)
                
                conversations.append([
                    {"role": "system", "content": ""},
                    {"role": "user", "content": f"{INSTRUCT_PROMPT} {prompt}"},
                    {"role": "assistant", "content": item['article']}
                ])
                self.answer_index.append(len(self.tokenizer.encode(prompt_only)))

            formatted_articles = [
                self.tokenizer.apply_chat_template(
                    conv,
                    tokenize=False,
                    add_generation_prompt=True
                ) for conv in conversations
            ]
            
            # Get unpadded lengths
            no_pad_lengths =  [min(self.max_length, len(self.tokenizer.encode(article))) for article in formatted_articles]
            
            # Do padded encoding
            self.encoded_articles = self.tokenizer(
                formatted_articles,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_offsets_mapping=True,
                return_tensors='pt'
            )
            
            # Calculate padding lengths and adjust indices
            pad_lengths = [self.encoded_articles.input_ids.size(1) - orig_len for orig_len in no_pad_lengths]
            self.answer_index = [idx + pad_len for idx, pad_len in zip(self.answer_index, pad_lengths)]
        else:
            # Original tokenization for non-instruct mode
            self.encoded_articles = self.tokenizer(
                [item['article'] for item in self.dataset],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_offsets_mapping=True,
                return_tensors='pt'
            )
            [self.answer_index.append(0) for _ in range(len(self.dataset))]
        
        # generate the unmemorize mask
        self.target_logits = []
        self.unmemorize_mask = []
        
        # always get the mask
        if unmemorize == True:
            self._apply_unmemorize()
            self._get_target_logits()
        
        # Tokenize all questions and answers     
        self.has_mcq = False  
        if 'mcq_questions' in self.dataset[0]:
            self.has_mcq = True
            self.encoded_questions = []
            self.encoded_options = []
            self.encoded_answers = []            
            for item in self.dataset:
                mcq = json.loads(item['mcq_questions'])
                questions = [q['question'] for q in mcq]
                options = [q['options'] for q in mcq]
                options_flat = [f'A:{d["A"]}' + '\n' + f'B:{d["B"]}' + '\n' + f'C:{d["C"]}' + '\n' + f'D:{d["D"]}' + '\n\nThe answer is letter' for d in options]
                answers = [q['correct_answer'] for q in mcq]
                
                # concatenate the questions and options
                questions_and_options = [f"{q}\n{a}" \
                                    for q, a in zip(questions, options_flat)]

                if self.instruct:
                    # Format questions as chat conversations
                    q_conversations = [
                        [{"role": "user", "content": q}] 
                        for q in questions_and_options
                    ]
                    formatted_questions = [
                        self.tokenizer.apply_chat_template(
                            conv,
                            tokenize=False,
                            add_generation_prompt=True
                        ) for conv in q_conversations
                    ]
                    encoded_q = self.tokenizer(
                        formatted_questions,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                else:
                    encoded_q = self.tokenizer(
                        questions_and_options,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                
                # Options and answers don't need chat format
                encoded_options = self.tokenizer(
                    options_flat,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,    
                    return_tensors='pt'
                )
                encoded_a = self.tokenizer(
                    answers,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                self.encoded_questions.append(encoded_q)
                self.encoded_options.append(encoded_options)
                self.encoded_answers.append(encoded_a)
                                    
    def _get_target_logits(self):
        """
        Optimized version of _get_target_logits that pre-computes token properties
        for the vocabulary to avoid repeated decoding, and preserves original position indexing.
        Assumes self.target_logits is initialized as an empty list before calling.
        """
        BATCH_SIZE = 32 # Adjust based on memory constraints
        encoded_input_ids = self.encoded_articles['input_ids']
        encoded_attention_mask = self.encoded_articles['attention_mask']
        unmemorize_masks = self.encoded_articles['unmemorize_mask']
        total_samples = len(encoded_input_ids)

        if total_samples == 0:
            return # Nothing to process

        # Determine processing device (use model's device if available, else input tensor's device)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device) # Move model to device
        # try:
        #     device = next(self.model.parameters()).device
        # except Exception:
        #     device = encoded_input_ids.device if isinstance(encoded_input_ids, torch.Tensor) else torch.device("cpu")

        self.model.eval() # Ensure model is in evaluation mode

        # --- Pre-computation of Token Properties ---
        # Define special tokens to skip based on tokenizer attributes
        skip_tokens = set()
        for attr in ['pad_token_id', 'eos_token_id', 'bos_token_id', 'sep_token_id', 'cls_token_id']:
            token_id = getattr(self.tokenizer, attr, None)
            if token_id is not None:
                skip_tokens.add(token_id)

        # Pre-compute properties for all tokens in the vocabulary
        precomputed_token_properties = {}
        vocab_size = self.tokenizer.vocab_size
        # Optional: Add tqdm progress bar for pre-computation if vocab_size is large
        vocab_iterator = range(vocab_size)
        # if tqdm and vocab_size > 10000: # Example threshold
        #    vocab_iterator = tqdm(vocab_iterator, desc="Pre-computing token properties", leave=False)

        for token_id in vocab_iterator:
            # Handle potential None token_id if vocab isn't dense (unlikely for HF tokenizers)
            if token_id is None: continue

            # Basic check for obviously invalid IDs (though vocab_size should be accurate)
            if not isinstance(token_id, int) or token_id < 0: continue

            try:
                # Decode the single token ID
                # Use clean_up_tokenization_spaces=False to preserve leading spaces accurately
                token_text = self.tokenizer.decode([token_id],
                                                   skip_special_tokens=False,
                                                   clean_up_tokenization_spaces=False)
            except Exception:
                 # Handle rare cases where decoding might fail for specific internal/unused IDs
                 token_text = ""

            is_special = token_id in skip_tokens

            # Check for leading space (common in SentencePiece/BPE)
            # Use the decoded text directly as startswith(' ') might miss symbols like 'Ġ'
            # A more robust check might involve tokenizer-specific logic if available
            # For now, stick to checking the decoded string's start
            has_space = token_text.startswith(' ') # Common heuristic, might need refinement

            # Check capitalization (after removing potential leading space)
            stripped_text = token_text.lstrip(' ') # Only strip leading space
            is_capitalized = len(stripped_text) > 0 and stripped_text[0].isupper()

            precomputed_token_properties[token_id] = {
                # 'text': token_text, # Store text only if needed later, otherwise skip to save memory
                'is_special': is_special,
                'has_space': has_space,
                'is_capitalized': is_capitalized
            }
        # --- End of Pre-computation ---


        # Setup batch processing and progress bar
        num_batches = (total_samples + BATCH_SIZE - 1) // BATCH_SIZE
        use_progress = total_samples > 1
        if use_progress and tqdm:
            batch_iterator = tqdm(range(num_batches), desc="Getting target logits")
        else:
            batch_iterator = range(num_batches)

        # Process samples in batches
        for batch_idx in batch_iterator:
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, total_samples)

            # Get batch data and move to device
            batch_input_ids = encoded_input_ids[start_idx:end_idx].to(device)
            batch_attention_mask = encoded_attention_mask[start_idx:end_idx].to(device)
            batch_unmemorize_masks = unmemorize_masks[start_idx:end_idx].cpu() # Keep masks on CPU for easier indexing

            # Forward pass to get logits
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    # labels are not strictly needed if loss is not used, but passing them often doesn't hurt
                    # labels=batch_input_ids,
                )
                batch_logits = outputs.logits.cpu() # Move logits to CPU for processing loop

            # Process each sample in the batch
            for i in range(batch_logits.shape[0]): # Iterate over samples in the current batch
                sample_global_idx = start_idx + i
                sample_logits_tensor = batch_logits[i]
                unmemorize_mask = batch_unmemorize_masks[i]
                attention_mask = batch_attention_mask[i].cpu() # Ensure attention mask is on CPU for checks
                input_ids_sample = batch_input_ids[i].cpu() # Ensure input IDs are on CPU for checks

                sample_results = {'tokens': {}} # Store results for this sample

                # Find where the actual sequence starts (first 1 in attention mask)
                valid_positions = torch.nonzero(attention_mask != 0, as_tuple=True)[0]
                if len(valid_positions) == 0:
                    self.target_logits.append(sample_results) # Append empty results for empty sample
                    continue

                first_valid_pos = valid_positions[0].item()
                seq_len = len(input_ids_sample) # Use actual length

                # Process each token position in the sequence
                for pos_idx in range(first_valid_pos, seq_len):
                    # Skip positions beyond attention mask explicitly (redundant if loop max is seq_len, but safe)
                    if pos_idx >= len(attention_mask) or attention_mask[pos_idx] == 0:
                        continue

                    # Calculate probability distribution for the *next* token prediction
                    # Use float for softmax for numerical stability
                    response_probs = torch.softmax(sample_logits_tensor[pos_idx].float(), dim=-1)

                    # Check if the *next* token position is marked for unmemorization
                    check_unmemorize_idx = pos_idx + 1
                    apply_unmemorize_filter = (
                        check_unmemorize_idx < len(unmemorize_mask) and
                        unmemorize_mask[check_unmemorize_idx] == 1
                    )

                    if apply_unmemorize_filter:
                        # --- Unmemorize Filtering Logic ---
                        # Get the token we want to avoid predicting
                        article_token_id = input_ids_sample[check_unmemorize_idx].item()

                        # Get properties of the token to avoid using the precomputed dict
                        # Use .get for safety, providing default neutral properties if ID somehow missing
                        default_props = {'is_special': False, 'has_space': False, 'is_capitalized': False}
                        article_token_props = precomputed_token_properties.get(article_token_id, default_props)

                        # Get more candidates than needed initially
                        k_initial = min(6 * self.unmemorize_top_k, len(response_probs))
                        top_initial_probs, top_initial_tokens = torch.topk(response_probs, k=k_initial)

                        # Filter out the article token and apply smart selection if enabled
                        valid_indices = []
                        for j in range(len(top_initial_tokens)):
                            candidate_token_id = top_initial_tokens[j].item()

                            # Skip the token we want to avoid
                            if candidate_token_id == article_token_id:
                                continue

                            # Apply smart selection rules if enabled
                            if self.unmemorize_smart_select:
                                candidate_props = precomputed_token_properties.get(candidate_token_id, default_props)

                                # Rule 1: First alternative shouldn't be special
                                if len(valid_indices) == 0:
                                    if candidate_props['is_special']:
                                        continue

                                # Rule 2: Check space prefix consistency
                                if article_token_props['has_space'] and not candidate_props['has_space']:
                                    continue

                                # Rule 3: Check capitalization consistency
                                if article_token_props['is_capitalized'] != candidate_props['is_capitalized']:
                                    continue

                            # If checks pass, add index to list
                            valid_indices.append(j)
                            if len(valid_indices) >= self.unmemorize_top_k:
                                break # Found enough valid candidates

                        # --- Refetching logic (Simplified - using same initial pool for simplicity now) ---
                        # Original code had complex refetching; for simplicity here, we work with the initial pool.
                        # If the initial fetch (k*6) wasn't enough, the fallback will handle it.
                        # A more complex refetch could be added back if necessary.

                        # Select the final tokens based on valid_indices found
                        if len(valid_indices) >= self.unmemorize_top_k:
                            # We found enough during the first pass
                            final_indices = torch.tensor(valid_indices[:self.unmemorize_top_k], dtype=torch.long)
                            top_probs = top_initial_probs[final_indices]
                            top_tokens = top_initial_tokens[final_indices]
                        elif len(valid_indices) > 0:
                           # Found some, but fewer than k, use what we found
                           final_indices = torch.tensor(valid_indices, dtype=torch.long)
                           top_probs = top_initial_probs[final_indices]
                           top_tokens = top_initial_tokens[final_indices]
                        else:
                            # --- Fallback if no valid tokens found by filtering ---
                            # Take top k+1, remove the article token, take top k of remainder
                            k_fallback = min(self.unmemorize_top_k + 1, len(response_probs))
                            fallback_probs, fallback_tokens = torch.topk(response_probs, k=k_fallback)

                            # Create mask to exclude the article token
                            mask = fallback_tokens != article_token_id
                            top_tokens = fallback_tokens[mask][:self.unmemorize_top_k]
                            top_probs = fallback_probs[mask][:self.unmemorize_top_k]

                            # Handle edge case where article_token was the only token predicted
                            if len(top_tokens) == 0:
                                # What to do here? Maybe take top_k excluding article token even if it means fewer than k?
                                # Or take the absolute top_k even if it includes invalid ones?
                                # Let's just take the original top_k in this very rare case.
                                top_probs, top_tokens = torch.topk(response_probs, k=self.unmemorize_top_k)


                    else:
                        # --- Normal Token Logic (No Unmemorize Filter) ---
                        k_normal = min(self.unmemorize_top_k, len(response_probs))
                        top_probs, top_tokens = torch.topk(response_probs, k=k_normal)

                    # Normalize the final selected probabilities
                    if top_probs.sum() > 1e-6: # Avoid division by zero if all probs are tiny
                         top_probs = top_probs / top_probs.sum()
                    # else: leave as is (likely all zeros or near zeros)

                    # Store results keyed by the current position index
                    sample_results['tokens'][pos_idx] = {
                        'top_tokens': top_tokens.tolist(), # Convert to list for storage
                        'top_probs': top_probs.tolist(),   # Convert to list for storage
                    }

                # Append results for the processed sample
                self.target_logits.append(sample_results)
                                                                                                    
    def _get_token_info(self, encoding, word_index, outputs):
        """Get token info and probability for a given word index"""
        token_id = encoding[word_index]
        token = self.tokenizer.decode(token_id)
        prob = F.softmax(outputs.logits[0][word_index-1], dim=-1)
        return token_id, token, prob[token_id]
        
    def _find_valid_word_token(self, encoding, word_ids, word_index, outputs):
        while word_index > 0:
            _, token, _ = self._get_token_info(encoding, word_index, outputs)
            # Check if token is start of word (not punctuation/space)  
            if not re.match(r'^[^\w\s]', token[0]):
                word_id = word_ids[word_index]
                # And check we're at start of word
                if word_index == 0 or word_ids[word_index-1] != word_id:
                    break
            word_index -= 1
        return word_index

    def _create_word_ids(self, encoded_tokens):
        # Decode tokens back to text
        decoded_tokens = [self.tokenizer.decode([token]) for token in encoded_tokens]
        
        word_ids = []
        current_word_id = -1     
        
        for i, token in enumerate(decoded_tokens):
            # Skip if empty token
            if not token:
                word_ids.append(-1)
                continue
                
            # Check if token starts new word
            is_new_word = False
            
            # If first token
            if i == 0:
                is_new_word = True
            else:
                # Get first char of current token
                first_char = token[0]
                
                # Check if this is part of a contraction
                is_contraction = re.match(r"(?i:^'s|^'t|^'re|^'ve|^'m|^'ll|^'d)", token)
                
                # Check if it matches pattern for word start
                if not re.match(r'[^\w]', first_char):
                    # It's a letter/number - check if previous token ended a word
                    prev_token = decoded_tokens[i-1]
                    if (not prev_token or prev_token[-1].isspace() or 
                        (re.match(r'[^\w\s]', prev_token[-1]) and not prev_token[-1] == "'")):
                        is_new_word = True
                        
                # If it's a contraction, keep same word_id
                elif is_contraction:
                    is_new_word = False
                # Special case for punctuation/spaces
                elif re.match(r'[^\w]', first_char):
                    is_new_word = True
                        
            if is_new_word:
                current_word_id += 1
                    
            word_ids.append(current_word_id)
        
        return word_ids
                    
    def _apply_unmemorize(self):
        """
        Optimized version of apply_unmemorize that processes articles in batches.
        Includes a highly optimized path for when smart_stride is disabled,
        avoiding unnecessary computations like word_ids and model inference.
        """
        # Initialize storage for the masks
        unmemorize_mask_list = []

        # Get input data references
        encoded_input_ids = self.encoded_articles['input_ids']
        encoded_attention_mask = self.encoded_articles['attention_mask']
        num_articles = len(encoded_input_ids)

        # Handle empty input case
        if num_articles == 0:
            # Determine expected shape based on how input_ids is stored (e.g., list of tensors or stacked tensor)
            max_len = encoded_input_ids.shape[1] if isinstance(encoded_input_ids, torch.Tensor) and encoded_input_ids.ndim == 2 else 0
            # If it's a list, need to handle differently or ensure consistency upstream
            # Assuming input_ids is a Tensor [num_articles, seq_len]
            self.encoded_articles['unmemorize_mask'] = torch.empty((0, max_len), dtype=torch.int64)
            return

        # Determine processing device (use model's device if available, else input tensor's device)
        try:
            device = next(self.model.parameters()).device
        except Exception: # Handle cases where model has no parameters or is not standard nn.Module
            device = encoded_input_ids.device if isinstance(encoded_input_ids, torch.Tensor) else torch.device("cpu")

        BATCH_SIZE = 32  # Adjust based on memory constraints

        # --- Conditional Model Inference ---
        # Only run inference if smart stride is enabled, as it needs logits.
        all_logits = None
        if self.unmemorize_smart_stride:
            print("Running model inference for smart stride...")
            all_logits_list = []
            self.model.eval() # Ensure model is in evaluation mode
            with torch.no_grad():
                batch_iterator_inf = range(0, num_articles, BATCH_SIZE)
                if tqdm and num_articles > BATCH_SIZE: # Show progress only if multiple batches
                   batch_iterator_inf = tqdm(batch_iterator_inf, desc="Model Inference", leave=False)

                for batch_start in batch_iterator_inf:
                    batch_end = min(batch_start + BATCH_SIZE, num_articles)
                    # Ensure batch tensors are on the correct device for the model
                    batch_ids = encoded_input_ids[batch_start:batch_end].to(device)
                    batch_attention = encoded_attention_mask[batch_start:batch_end].to(device)

                    outputs = self.model(
                        input_ids=batch_ids,
                        attention_mask=batch_attention,
                        # No labels needed if only using logits for inference
                    )
                    # Move logits to CPU for potentially large datasets to avoid GPU OOM during aggregation/processing
                    all_logits_list.append(outputs.logits.cpu())

            if all_logits_list:
                all_logits = torch.cat(all_logits_list, dim=0)
            del all_logits_list # Free memory


        # --- Pre-calculate Word IDs (only if needed) ---
        all_word_ids = None
        if self.unmemorize_smart_stride:
            print("Calculating word IDs for smart stride...")
            # Assuming _create_word_ids operates on a single encoding tensor
            # and returns a tensor/list. Processing on CPU assumed.
            word_id_iterator = range(num_articles)
            if tqdm and num_articles > 1: # Show progress
                 word_id_iterator = tqdm(word_id_iterator, desc="Computing Word IDs", leave=False)
            # Make sure input_ids are accessible, potentially move to CPU if needed by _create_word_ids
            input_ids_cpu = encoded_input_ids.cpu() if isinstance(encoded_input_ids, torch.Tensor) else encoded_input_ids
            all_word_ids = [self._create_word_ids(input_ids_cpu[idx]) for idx in word_id_iterator]


        # --- Mask Generation Loop ---
        print("Generating unmemorize masks...")
        batch_iterator_mask = range(0, num_articles, BATCH_SIZE)
        if tqdm and num_articles > BATCH_SIZE: # Show progress
            desc = "Applying unmemorize (Smart)" if self.unmemorize_smart_stride else "Applying standard unmemorize"
            batch_iterator_mask = tqdm(batch_iterator_mask, desc=desc)

        # Ensure data used in the loop is on CPU to avoid per-item GPU overhead if logic is complex
        # Or keep on GPU if loops are simple and data transfer is the bottleneck. CPU is safer for Python loops.
        input_ids_cpu = encoded_input_ids.cpu() if isinstance(encoded_input_ids, torch.Tensor) else encoded_input_ids
        attention_mask_cpu = encoded_attention_mask.cpu() if isinstance(encoded_attention_mask, torch.Tensor) else encoded_attention_mask


        for batch_start in batch_iterator_mask:
            batch_end = min(batch_start + BATCH_SIZE, num_articles)
            batch_size = batch_end - batch_start

            # Get data for the current batch (already on CPU)
            batch_ids_cpu = input_ids_cpu[batch_start:batch_end]
            batch_attention_cpu = attention_mask_cpu[batch_start:batch_end]
            batch_logits_cpu = all_logits[batch_start:batch_end] if all_logits is not None else None
            batch_word_ids_cpu = all_word_ids[batch_start:batch_end] if all_word_ids is not None else None


            for i in range(batch_size):
                article_idx_global = batch_start + i
                encoded_article = batch_ids_cpu[i]
                attention_mask = batch_attention_cpu[i]
                article_len = len(encoded_article) # Get length once

                # Initialize mask (on CPU)
                unmemorize_mask = torch.zeros_like(attention_mask, dtype=torch.int64)

                # Find first actual token index (skip padding/CLS)
                non_zero_indices = torch.nonzero(attention_mask != 0, as_tuple=True)[0]
                if len(non_zero_indices) == 0:
                    unmemorize_mask_list.append(unmemorize_mask)
                    continue # Skip empty sequences

                start_index = non_zero_indices[0].item() + 1 # Adjust +1 based on tokenization strategy

                # --- Apply Standard or Smart Logic ---
                if not self.unmemorize_smart_stride:
                    # --- >>> Optimized Standard Path <<< ---
                    # No word_ids, logits, softmax, decode, regex needed.
                    current_index = start_index + self.unmemorize_start

                    while current_index < article_len and attention_mask[current_index] != 0:
                        # Apply mask across the span efficiently
                        span_end_index = min(current_index + self.unmemorize_span, article_len)
                        for token_pos in range(current_index, span_end_index):
                            if attention_mask[token_pos] != 0:
                                unmemorize_mask[token_pos] = 1
                            else:
                                # Hit padding within the span, stop processing this span
                                break
                        # Move to the next stride position
                        current_index += self.unmemorize_stride
                    # --- >>> End of Non-Smart Path <<< ---

                else:
                    # --- Smart Path (Requires word_ids and logits) ---
                    if batch_word_ids_cpu is None or batch_logits_cpu is None:
                         # Should not happen if logic is correct, but defensive check
                         raise ValueError("Word IDs or Logits not available for smart stride.")

                    word_ids = batch_word_ids_cpu[i] # Get pre-calculated word IDs for this article
                    logits = batch_logits_cpu[i]     # Get pre-calculated logits for this article

                    # Pre-calculate probabilities for the entire article once
                    try:
                        # Ensure logits are float for softmax
                        article_probs = F.softmax(logits.float(), dim=-1)
                    except Exception as e:
                        print(f"Error during softmax for article {article_idx_global}: {e}")
                        # Decide how to handle: skip article, use default probs, etc.
                        # For now, skip processing this article's smart logic
                        unmemorize_mask_list.append(unmemorize_mask) # Append zero mask
                        continue

                    current_index = start_index + self.unmemorize_start

                    while current_index < article_len and attention_mask[current_index] != 0:
                        for span_offset in range(self.unmemorize_span):
                            candidate_index = current_index + span_offset

                            if candidate_index >= article_len or attention_mask[candidate_index] == 0:
                                break # Span goes out of bounds or into padding

                            # --- Smart Logic Search (backward from candidate) ---
                            target_mask_index = -1 # Reset for each candidate in span
                            search_idx = candidate_index
                            while search_idx >= start_index: # Search backward
                                current_word_id = word_ids[search_idx] # Assumes word_ids is indexable
                                is_word_start = (search_idx == start_index or word_ids[search_idx - 1] != current_word_id)

                                if is_word_start:
                                    token_id = encoded_article[search_idx].item()

                                    # Check probability from pre-calculated tensor
                                    # Probability of current token (at search_idx) predicted by previous token (search_idx - 1)
                                    prob = 0.0
                                    if search_idx > 0:
                                         # Ensure indices are valid before accessing probs
                                         if (search_idx - 1) < article_probs.shape[0] and token_id < article_probs.shape[1]:
                                             prob = article_probs[search_idx - 1, token_id].item()
                                         else:
                                             print(f"Warning: Index out of bounds accessing probs for article {article_idx_global}, index {search_idx-1}, token {token_id}")


                                    # Check punctuation (still inefficient part)
                                    # TODO: Replace with faster method if possible
                                    token = self.tokenizer.decode([token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                    # Check if token is non-empty and does *not* start with common non-alphanumeric (excluding whitespace)
                                    # This regex might need refinement based on tokenizer specifics
                                    is_not_punct_start = token and not re.match(r'^[^\w\s]', token)


                                    if is_not_punct_start and prob < 1.0:
                                        target_mask_index = search_idx
                                        break # Found suitable token, stop backward search

                                search_idx -= 1 # Continue searching backward

                            # Mask if the smart logic found a valid target within the search
                            if target_mask_index != -1:
                                unmemorize_mask[target_mask_index] = 1
                            # else: If no suitable token found by smart logic for this span_offset, do nothing

                        # Move to the next stride position
                        current_index += self.unmemorize_stride
                    # --- End of Smart Path ---

                # Append the final mask for this article
                unmemorize_mask_list.append(unmemorize_mask)

        # --- Final Stacking ---
        # Stack the list of mask tensors into a single tensor
        if unmemorize_mask_list:
            try:
                final_mask_tensor = torch.stack(unmemorize_mask_list)
            except RuntimeError as e:
                # This can happen if masks have different lengths, indicating an issue upstream (padding?)
                print(f"Error stacking masks: {e}. Check input padding consistency.")
                # Handle error - perhaps return the list or raise exception
                raise e
            self.encoded_articles['unmemorize_mask'] = final_mask_tensor
        else:
            # Handle case where input was non-empty but generated no masks (e.g., all empty sequences filtered out)
             max_len = encoded_input_ids.shape[1] if isinstance(encoded_input_ids, torch.Tensor) and encoded_input_ids.ndim == 2 else 0

                
    def tokenized_article(self, idx):
        return self.tokenized_articles[idx]
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        # Get pre-tokenized article
        article_ids = self.encoded_articles['input_ids'][idx]
        article_mask = self.encoded_articles['attention_mask'][idx]

        # get target logits
        if self.target_logits:
            target_logits = self.target_logits[idx]
        else:
            # provide value that's not None
            target_logits = {'tokens': {}}
            
        # check if it has an unmemorize mask key in the dictionary
        if self.encoded_articles.get('unmemorize_mask') is not None: 
            unmemorize_mask = self.encoded_articles['unmemorize_mask'][idx]
        else:
            # provide value that's not None
            unmemorize_mask = torch.zeros_like(article_mask)
        
        # Get pre-tokenized questions and answers
        if self.has_mcq:
            q_ids = self.encoded_questions[idx]['input_ids']
            q_mask = self.encoded_questions[idx]['attention_mask']
            a_ids = self.encoded_answers[idx]['input_ids']
            a_mask = self.encoded_answers[idx]['attention_mask']
            o_ids = self.encoded_options[idx]['input_ids']
            o_mask = self.encoded_options[idx]['attention_mask']
            
            return {
                'article_ids': article_ids,
                'article_mask': article_mask,
                'unmemorize_mask': unmemorize_mask,
                'target_logits': target_logits,
                'answer_index': self.answer_index[idx],
                'q_ids': q_ids,
                'q_mask': q_mask,
                'a_ids': a_ids,
                'a_mask': a_mask,
                'o_ids': o_ids,
                'o_mask': o_mask,
            }   
        else:
            return {
                'article_ids': article_ids,
                'article_mask': article_mask,
                'unmemorize_mask': unmemorize_mask,
                'target_logits': target_logits,
                'answer_index': self.answer_index[idx]
            }
        

def get_unmemorize_probabilities(logits, labels, attention_mask, unmemorize_mask):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()    
    if unmemorize_mask is not None:
        shift_unmemorize_mask = unmemorize_mask[..., 1:].contiguous()
    else:
        shift_unmemorize_mask = attention_mask[..., :-1].contiguous()
    
    # Get the vocabulary size from the logits
    vocab_size = logits.size(-1)
    
    # Apply softmax to get probabilities
    probabilities = F.softmax(shift_logits, dim=2)
    
    # Flatten the tensors for easier indexing
    flat_probabilities = probabilities.view(-1, vocab_size)
    flat_input_ids = shift_labels.view(-1)
    flat_unmemorize_mask = shift_unmemorize_mask.view(-1)
    
    # Get the indices where unmemorize_mask is True
    unmemorize_indices = flat_unmemorize_mask.nonzero().squeeze()
    
    # Get the corresponding input_ids and probabilities
    target_input_ids = flat_input_ids[unmemorize_indices]
    target_probabilities = flat_probabilities[unmemorize_indices]
    
    # Extract the probabilities of the target tokens
    result_probabilities = target_probabilities[torch.arange(len(target_input_ids)), target_input_ids]
    
    # Set probabilities of 1.0 to 0.0 (with small epsilon for floating point comparison)
    if unmemorize_mask is not None:
        epsilon = 1e-15
        result_probabilities = torch.where(
            (result_probabilities >= 1.0 - epsilon), 
            torch.zeros_like(result_probabilities),
            result_probabilities
        )
    
    return result_probabilities
        
def calculate_kl_loss(logger, model, tokenizer, outputs, labels, attention_mask, target_logits, unmemorize_mask, debug = False):
    logits = outputs[..., :-1, :].contiguous()  # Shape: [batch_size, seq_len-1, vocab_size]
    shift_labels = labels[..., 1:].contiguous()       
    shift_attention_mask = attention_mask[..., 1:].contiguous()
    shift_unmemorize_mask = unmemorize_mask[..., 1:].contiguous()
    
    # Initialize total loss
    total_kl_loss = 0.0
    batch_size = logits.size(0)
    target_logit_idx = 0
    
    for batch_idx in range(batch_size):
        # Find where the actual sequence starts (first 1 in attention mask)
        valid_tokens = shift_attention_mask[batch_idx].nonzero().squeeze(-1)
        if len(valid_tokens) == 0:
            continue
        
        # Process only the valid token positions
        for pos in valid_tokens:
            # Get predicted distribution for current position
            pred_logits = logits[batch_idx, pos]
            pred_probs = torch.softmax(pred_logits, dim=-1)
                
            top_tokens_tensor = target_logits['tokens'][pos.item()]['top_tokens']
            top_probs_tensor = target_logits['tokens'][pos.item()]['top_probs']
            
            # Construct the vectors for this batch index
            top_tokens = torch.tensor([tensor[batch_idx].item() for tensor in top_tokens_tensor], device=model.device)
            top_probs = torch.tensor([tensor[batch_idx].item() for tensor in top_probs_tensor], 
                                            device=model.device, dtype=pred_probs.dtype)            
            # Create target distribution tensor
            target_dist = torch.zeros_like(pred_probs)
            target_dist[top_tokens] = top_probs.clone().detach()

            if debug == True and batch_idx == 0 and pos.item()-valid_tokens[0]:
                # get top predicted token 
                top_pred_token = torch.argmax(pred_probs)
                
                # get top target token
                top_target_token = top_tokens[0]
                
                if top_pred_token == top_target_token:
                    logger.info(f"{pos.item()-valid_tokens[0]-1}: {tokenizer.decode([top_pred_token])}")
                else:
                    logger.info(f"{pos.item()-valid_tokens[0]-1}: *** {tokenizer.decode([top_pred_token])} vs {tokenizer.decode([top_target_token])}")

            # Calculate KL divergence for this position
            if pos.item()-valid_tokens[0]:
                token_kl = F.kl_div(
                    torch.log(pred_probs + 1e-10),  # Add small epsilon to avoid log(0)
                    target_dist,
                    reduction='sum'
                )
            else:
                token_kl = 0.0
                            
            # get probabilities of label ids for unmemorize mask tokens 
            if shift_unmemorize_mask[batch_idx, pos] == 1:
                # Get predicted probability for the target token
                target_token = shift_labels[batch_idx, pos]
                target_prob = pred_probs[target_token]
                
                # Add negative log probability to KL loss
                token_kl = target_prob * 100
                
                # if the target probability is less than 0.01, no need
                # to push it further down
                if target_prob < 0.01:
                    token_kl = 0
                    
                # if target prob is 1.0, then leave it since it won't budge
                if target_prob == 1.0:
                    if debug == True:
                        logger.info(f"   [{pos-1}] Target Prob is 1.0 - skipping")
                    token_kl = 0

                if debug == True and batch_idx == 0 and pos.item()-valid_tokens[0]:
                    logger.info(f"   Unmemorize Loss: {token_kl}")                

            else:
                if debug == True and batch_idx == 0 and pos.item()-valid_tokens[0]:
                    logger.info(f"   Loss: {token_kl}")                
            
            total_kl_loss += token_kl
            target_logit_idx += 1
    
    # Normalize by total number of tokens
    total_tokens = shift_attention_mask.sum()
    if total_tokens > 0:
        total_kl_loss = total_kl_loss / total_tokens
    
    if debug == True:            
        logger.info(f"   Total Unmemorize Loss: {total_kl_loss}")                      
    return total_kl_loss
