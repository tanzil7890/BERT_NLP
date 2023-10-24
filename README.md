# Project 1: BERT Question Answering Model for SQuAD

This module is designed to use a pre-trained BERT model for the SQuAD (Stanford Question Answering Dataset) task. It accepts a context paragraph and a question, and returns the answer by selecting a span from the given context.

## Functions:

### 1. `tokenize_context(text_words)`

- **Input**: A list of words, ideally returned by `whitespace_split()`.
- **Output**: Tokens for each word and a list of word_ids for each token to map back to the original word.

### 2. `get_ids(tokens)`

- **Input**: List of tokens.
- **Output**: Converts each token to its corresponding ID in the BERT tokenizer.

### 3. `get_mask(tokens)`

- **Input**: List of tokens.
- **Output**: Creates a mask to differentiate actual tokens from padding tokens.

### 4. `get_segments(tokens)`

- **Input**: List of tokens.
- **Output**: Creates segment ids to differentiate between the question and context.

### 5. `create_input_dict(question, context)`

- **Input**: A question and a context as strings.
- **Output**: A dictionary containing token ids, mask, and segment ids needed for the model. Also returns token to word mapping, and other meta data.

## Workflow:

1. Set the context and the question.
2. Process the input to get tokens, ids, masks, and segment ids.
3. Predict the start and end tokens of the answer using the BERT model.
4. Interpret the result to extract the answer span.
5. Highlight the answer in the context and display.

## Example:

Given the context:
> "Neoclassical economics views inequalities in the distribution of income as arising from differences in value added by labor, capital and land."

And the question:
> "What are examples of economic actors?"

The answer is extracted as:
> "worker, capitalist/business owner, landlord"

The answer is also highlighted in the original context for clarity.

## Dependencies:

- TensorFlow
- NumPy
- BERT Tokenizer

## How to use:

1. Load your pre-trained BERT model.
2. Call the `create_input_dict` function with your question and context.
3. Use the BERT model to predict the start and end logits.
4. Extract and highlight the answer using provided functions.

## Note:

This module assumes a maximum input length of 384 tokens for the BERT model.
