# TokenMonster Documentation

## Table of Contents
1. [Understanding the Purpose](#understanding-the-purpose)
2. [Overview and Introduction](#overview-and-introduction)
3. [Class Definition](#class-definition)
4. [Functionality and Usage](#functionality-and-usage)
    - [Initializing TokenMonster](#initializing-tokenmonster)
    - [Setting Local Directory](#setting-local-directory)
    - [Loading Vocabulary](#loading-vocabulary)
    - [Creating a New Vocabulary](#creating-a-new-vocabulary)
    - [Saving Vocabulary](#saving-vocabulary)
    - [Exporting Vocabulary to YAML](#exporting-vocabulary-to-yaml)
    - [Tokenization](#tokenization)
    - [Decoding Tokens](#decoding-tokens)
    - [Creating a Decoder Instance](#creating-a-decoder-instance)
    - [Getting Vocabulary Dictionary](#getting-vocabulary-dictionary)
    - [Getting Character Set](#getting-character-set)
    - [Getting Normalization](#getting-normalization)
    - [Getting Capcode Level](#getting-capcode-level)
    - [Getting Optimization Mode](#getting-optimization-mode)
    - [Mapping Token ID to Token String](#mapping-token-id-to-token-string)
    - [Mapping Token ID to Token String (Decoded)](#mapping-token-id-to-token-string-decoded)
    - [Mapping Token String to Token ID](#mapping-token-string-to-token-id)
    - [Modifying Vocabulary](#modifying-vocabulary)
    - [Adding Regular Tokens](#adding-regular-tokens)
    - [Deleting Tokens](#deleting-tokens)
    - [Deleting Tokens by ID](#deleting-tokens-by-id)
    - [Adding Special Tokens](#adding-special-tokens)
    - [Resizing Vocabulary](#resizing-vocabulary)
    - [Resetting Token IDs](#resetting-token-ids)
    - [Enabling UNK Token](#enabling-unk-token)
    - [Disabling UNK Token](#disabling-unk-token)
    - [Disconnecting TokenMonster](#disconnecting-tokenmonster)
    - [Serializing Tokens](#serializing-tokens)
    - [Deserializing Tokens](#deserializing-tokens)
5. [Additional Information](#additional-information)
6. [Examples](#examples)
7. [Conclusion](#conclusion)

## 1. Understanding the Purpose <a name="understanding-the-purpose"></a>

TokenMonster is a Python library designed to provide tokenization and vocabulary management functionalities. It allows you to tokenize text, manage vocabularies, modify vocabularies, and perform various operations related to tokenization and vocabulary handling.

### Purpose and Functionality

TokenMonster serves the following purposes and functionalities:

- **Tokenization**: Tokenize text into tokens based on a specified vocabulary.
- **Vocabulary Management**: Load, create, save, and modify vocabularies.
- **Token ID Mapping**: Map tokens to token IDs and vice versa.
- **Serialization**: Serialize and deserialize tokens for storage or transmission.
- **Configuration**: Access and modify vocabulary settings like character set, normalization, capcode level, and optimization mode.
- **Special Token Handling**: Add, delete, or modify special tokens.
- **Disconnecting**: Gracefully disconnect from TokenMonster server.

TokenMonster is useful in various natural language processing tasks, especially when working with custom vocabularies and tokenization requirements.

## 2. Overview and Introduction <a name="overview-and-introduction"></a>

### Overview

TokenMonster is a versatile library for tokenization and vocabulary management. It allows you to create, load, and modify vocabularies, tokenize text, and perform various operations related to tokenization.

### Importance and Relevance

In the field of natural language processing, tokenization is a fundamental step in text preprocessing. TokenMonster provides a flexible and efficient way to tokenize text while also enabling users to manage custom vocabularies. The ability to add, delete, or modify tokens is crucial when working with specialized language models and text data.

### Key Concepts and Terminology

Before diving into the details, let's clarify some key concepts and terminology used throughout the documentation:

- **Tokenization**: The process of breaking text into individual tokens (words, subwords, or characters).
- **Vocabulary**: A collection of tokens and their corresponding token IDs.
- **Token ID**: A unique identifier for each token in the vocabulary.
- **Special Tokens**: Tokens that have a specific role, such as padding, start of sentence, end of sentence, and unknown tokens.
- **Normalization**: Text processing operations like lowercasing, accent removal, and character set transformation.
- **Capcode Level**: The level of capcoding applied to tokens (0-2).
- **Optimization Mode**: The mode used for optimizing TokenMonster (0-5).

Now that we have an overview, let's proceed with a detailed class definition.

## 3. Class Definition <a name="class-definition"></a>

### Class: TokenMonster

The `TokenMonster` class encapsulates the functionality of the TokenMonster library.

#### Constructor

```python
def __init__(self, path):
    """
    Initializes the TokenMonster class and loads a vocabulary.

    Args:
        path (str): A filepath, URL, or pre-built vocabulary name.
    """
```

### Methods

The `TokenMonster` class defines various methods to perform tokenization, vocabulary management, and configuration. Here are the key methods:

#### 1. Setting Local Directory

```python
def set_local_directory(self, dir=None):
    """
    Sets the local directory for TokenMonster.

    Args:
        dir (str, optional): The local directory to use. Defaults to None.
    """
```

#### 2. Loading Vocabulary

```python
def load(self, path):
    """
    Loads a TokenMonster vocabulary from file, URL, or by name.

    Args:
        path (str): A filepath, URL, or pre-built vocabulary name.
    """
```

#### 3. Loading Vocabulary (Multiprocess Safe)

```python
def load_multiprocess_safe(self, path):
    """
    Loads a TokenMonster vocabulary from file, URL, or by name. It's safe for multiprocessing,
    but vocabulary modification is disabled, and tokenization is slightly slower.

    Args:
        path (str): A filepath, URL, or pre-built vocabulary name.
    """
```

#### 4. Creating a New Vocabulary

```python
def new(self, yaml):
    """
    Creates a new vocabulary from a YAML string.

    Args:
        yaml (str): The YAML file.
    """
```

#### 5. Saving Vocabulary

```python
def save(self, fname):
    """
    Saves the current vocabulary to a file.

    Args:
        fname (str): The filename to save the vocabulary to.
    """
```

#### 6. Exporting Vocabulary to YAML

```python
def export_yaml(self, order_by_score=False):
    """
    Exports the vocabulary as a YAML file, which is returned as a bytes string.

    Args:
        order_by_score (bool, optional): If true, the tokens are ordered by score instead of alphabetically. Defaults to False.

    Returns:
        bytes: The vocabulary in YAML format.
    """
```

#### 7. Tokenization

```python
def tokenize(self, text):
    """
       Tokenizes a

    string into tokens according to the vocabulary.

       Args:
           text (str): A string or bytes string or a list of strings or bytes strings.

       Returns:
           numpy array: The token IDs.
    """
```

#### 8. Tokenization (Count)

```python
def tokenize_count(self, text):
    """
    Same as tokenize, but it returns only the number of tokens.

    Args:
        text (str): A string or bytes string or a list of strings or bytes strings.

    Returns:
        int: The number of tokens for each input string.
    """
```

#### 9. Decoding Tokens

```python
def decode(self, tokens):
    """
    Decodes tokens into a string.

    Args:
        tokens (int, list of int, or numpy array): The tokens to decode into a string.

    Returns:
        str: The composed string from the input tokens.
    """
```

#### 10. Creating a Decoder Instance

```python
def decoder(self):
    """
    Returns a new decoder instance used for decoding tokens into text.

    Returns:
        tokenmonster.DecoderInstance: A new decoder instance.
    """
```

#### 11. Getting Vocabulary Dictionary

```python
def get_dictionary(self):
    """
    Returns a dictionary of all tokens in the vocabulary.

    Returns:
        list: A list of dictionaries where the index is the token ID, and each is a dictionary.
    """
```

#### 12. Getting Character Set

```python
def charset(self):
    """
    Returns the character set used by the vocabulary.

    Returns:
        str: The character set used by the vocabulary. Possible values are "UTF-8" or "None".
    """
```

#### 13. Getting Normalization

```python
def normalization(self):
    """
    Returns the normalization of the vocabulary.

    Returns:
        str: The normalization of the vocabulary. Possible values are "None", "NFD", "Lowercase", "Accents", "Quotemarks", "Collapse", "Trim", "LeadingSpace", or "UnixLines".
    """
```

#### 14. Getting Capcode Level

```python
def capcode(self):
    """
    Returns the capcode level of the vocabulary.

    Returns:
        int: The capcode level (0-2).
    """
```

#### 15. Getting Optimization Mode

```python
def mode(self):
    """
    Returns the optimization mode of the vocabulary.

    Returns:
        int: The optimization mode (0-5).
    """
```

#### 16. Mapping Token ID to Token String

```python
def id_to_token(self, id):
    """
    Get the token string from a single token ID, in its capcode-encoded form.

    Args:
        id (int): The token ID.

    Returns:
        str or None: The token string corresponding to the input ID. None if the ID is not in the vocabulary.
    """
```

#### 17. Mapping Token ID to Token String (Decoded)

```python
def id_to_token_decoded(self, id):
    """
    Get the token string from a single token ID, in its capcode-decoded form.

    Args:
        id (int): The token ID.

    Returns:
        str or None: The token string corresponding to the input ID. None if the ID is not in the vocabulary.
    """
```

#### 18. Mapping Token String to Token ID

```python
def token_to_id(self, token):
    """
    Returns the ID of a single token.

    Args:
        token (str): The token to get the ID for.

    Returns:
        int or None: The ID of the token. None if the token is not in the vocabulary.
    """
```

#### 19. Modifying Vocabulary

```python
def modify(
    self,
    add_special_tokens=None,
    add_regular_tokens=None,
    delete_tokens=None,
    resize=None,
    change_unk=None,
):
    """
    Modifies the vocabulary.

    Args:
        add_special_tokens (str or list of str, optional): Special tokens to add to the vocabulary.
        add_regular_tokens (str or list of str, optional): Regular tokens to add to the vocabulary.
        delete_tokens (str or list of str, optional): Regular or special tokens to delete.
        resize (int, optional): Resizes the vocabulary to this size.
        change_unk (bool, optional): If set, it enables or disables the UNK token.

    Returns:
        int: The new size of the vocabulary.
    """
```

#### 20. Adding Regular Tokens

```python
def add_token(self, token):
    """
    Add one or more regular tokens.

    Args:
        token (str or list of str): The regular tokens to add.

    Returns:
        int: The new size of the vocabulary.
    """
```

#### 21. Deleting Tokens

```python
def delete_token(self, token):
    """
    Delete one or more regular or special tokens.

    Args:
        token (str or list of str): The tokens to delete.

    Returns:
        int: The new size of the vocabulary.
    """
```

#### 22. Deleting Tokens by ID

```python
def delete_token_by_id(self, id):
    """
    Delete one or more regular or special tokens by specifying the token ID.

    Args:
        id (int or list of int): The IDs of the tokens to delete.

    Returns:
        int: The new size of the vocabulary.
    """
```

#### 23. Adding Special Tokens

```python
def add_special_token(self, token):
    """
    Add one or more special tokens.

    Args:
        token (str or list of str): The special tokens to add.

    Returns:
        int: The new size of the vocabulary.
    """
```

#### 24. Resizing Vocabulary

```python
def resize(self, size):
    """
    Changes the size of the vocabulary.

    Args:
        size (int): The new size of the vocabulary.

    Returns:
        int: The new size of the vocabulary.
    """
```

#### 25. Resetting Token IDs

```python
def reset_token_ids(self):
    """
    Resets the token IDs to be sequential beginning from zero.
    """
```

#### 26. Enabling UNK Token

```python
def enable_unk_token(self):
    """
    Enables the UNK token.

    Returns:
        int: The new size of the vocabulary.
    """
```

#### 27. Disabling UNK Token

```python
def disable_unk_token(self):
    """
    Disables the UNK token.

    Returns:
        int: The new size of the vocabulary.
    """
```

#### 28. Disconnecting TokenMonster

```python
def disconnect(self):
    """
    Disconnects and closes TokenMonster server.
    """
```

#### 29. Serializing Tokens

```python
def serialize_tokens(self, integer_list):
    """
    Serializes tokens from a list of ints or numpy array into a binary string.

    Args:
        integer_list (list of int or numpy array): The tokens to serialize.

    Returns:
        bytes: The serialized binary string.
    """
```

#### 30. Deserializing Tokens

```python
def deserialize_tokens(self, binary_string):
    """
    Deserializes a binary string into a numpy array of token IDs.

    Args:


        binary_string (bytes): The binary string to deserialize.

    Returns:
        np.array: The deserialized tokens.
    """
```

This concludes the class definition. In the following sections, we will explore each method in detail and provide examples of their usage.

## 4. Functionality and Usage <a name="functionality-and-usage"></a>

### 4.1. Initializing TokenMonster <a name="initializing-tokenmonster"></a>

To get started with TokenMonster, you need to initialize an instance of the `TokenMonster` class. The constructor takes a single argument, `path`, which specifies the location of the vocabulary.

**Example:**

```python
from zeta.tokenizers import TokenMonster

# Initialize TokenMonster with a vocabulary file
tokenizer = TokenMonster("path/to/vocabulary")
```

### 4.2. Setting Local Directory <a name="setting-local-directory"></a>

You can set the local directory for TokenMonster using the `set_local_directory` method. This directory is used for local caching of vocabulary files.

**Example:**

```python
# Set the local directory for TokenMonster
tokenizer.set_local_directory("path/to/local/directory")
```

### 4.3. Loading Vocabulary <a name="loading-vocabulary"></a>

TokenMonster allows you to load vocabularies from various sources, including file paths, URLs, or pre-built vocabulary names. Use the `load` method to load a vocabulary.

**Example:**

```python
# Load a vocabulary from a file
tokenizer.load("path/to/vocabulary")
```

### 4.4. Creating a New Vocabulary <a name="creating-a-new-vocabulary"></a>

You can create a new vocabulary from a YAML string using the `new` method. This is useful when you want to define a custom vocabulary.

**Example:**

```python
# Create a new vocabulary from a YAML string
yaml_string = """
- token: [PAD]
  id: 0
"""
tokenizer.new(yaml_string)
```

### 4.5. Saving Vocabulary <a name="saving-vocabulary"></a>

TokenMonster allows you to save the current vocabulary to a file using the `save` method. This is useful for preserving custom vocabularies you've created.

**Example:**

```python
# Save the current vocabulary to a file
tokenizer.save("custom_vocabulary.yaml")
```

### 4.6. Exporting Vocabulary to YAML <a name="exporting-vocabulary-to-yaml"></a>

You can export the vocabulary as a YAML file using the `export_yaml` method. This method returns the vocabulary in YAML format as a bytes string.

**Example:**

```python
# Export the vocabulary to a YAML file
yaml_data = tokenizer.export_yaml()
```

### 4.7. Tokenization <a name="tokenization"></a>

Tokenization is a core functionality of TokenMonster. You can tokenize text into tokens according to the loaded vocabulary using the `tokenize` method.

**Example:**

```python
# Tokenize a text string
text = "Hello, world!"
token_ids = tokenizer.tokenize(text)
```

### 4.8. Tokenization (Count) <a name="tokenization-count"></a>

If you want to know the number of tokens without getting the token IDs, you can use the `tokenize_count` method.

**Example:**

```python
# Count the number of tokens in a text string
text = "Hello, world!"
token_count = tokenizer.tokenize_count(text)
```

### 4.9. Decoding Tokens <a name="decoding-tokens"></a>

To decode token IDs back into a human-readable string, you can use the `decode` method.

**Example:**

```python
# Decode token IDs into a string
decoded_text = tokenizer.decode(token_ids)
```

### 4.10. Creating a Decoder Instance <a name="creating-a-decoder-instance"></a>

TokenMonster allows you to create a decoder instance for decoding tokens into text. Use the `decoder` method to obtain a decoder instance.

**Example:**

```python
# Create a decoder instance
decoder = tokenizer.decoder()
```

### 4.11. Getting Vocabulary Dictionary <a name="getting-vocabulary-dictionary"></a>

The `get_dictionary` method returns a dictionary of all tokens in the vocabulary. Each dictionary entry contains information about the token.

**Example:**

```python
# Get the vocabulary dictionary
vocab_dict = tokenizer.get_dictionary()
```

### 4.12. Getting Character Set <a name="getting-character-set"></a>

You can retrieve the character set used by the vocabulary using the `charset` method.

**Example:**

```python
# Get the character set used by the vocabulary
charset = tokenizer.charset()
```

### 4.13. Getting Normalization <a name="getting-normalization"></a>

TokenMonster allows you to access the normalization settings applied to the vocabulary using the `normalization` method.

**Example:**

```python
# Get the normalization settings of the vocabulary
normalization = tokenizer.normalization()
```

### 4.14. Getting Capcode Level <a name="getting-capcode-level"></a>

The `capcode` method returns the capcode level of the vocabulary.

**Example:**

```python
# Get the capcode level of the vocabulary
capcode_level = tokenizer.capcode()
```

### 4.15. Getting Optimization Mode <a name="getting-optimization-mode"></a>

You can retrieve the optimization mode used for TokenMonster using the `mode` method.

**Example:**

```python
# Get the optimization mode of TokenMonster
optimization_mode = tokenizer.mode()
```

### 4.16. Mapping Token ID to Token String <a name="mapping-token-id-to-token-string"></a>

Given a token ID, you can use the `id_to_token` method to get the token string in its capcode-encoded form.

**Example:**

```python
# Get the token string from a token ID (capcode-encoded)
token_id = 42
token_string = tokenizer.id_to_token(token_id)
```

### 4.17. Mapping Token ID to Token String (Decoded) <a name="mapping-token-id-to-token-string-decoded"></a>

The `id_to_token_decoded` method is used to get the token string from a token ID in its capcode-decoded form.

**Example:**

```python
# Get the token string from a token ID (capcode-decoded)
token_id = 42
decoded_token_string = tokenizer.id_to_token_decoded(token_id)
```

### 4.18. Mapping Token String to Token ID <a name="mapping-token-string-to-token-id"></a>

You can obtain the token ID of a given token string using the `token_to_id` method.

**Example:**

```python
# Get the token ID from a token string
token_string = "apple"
token_id = tokenizer.token_to_id(token_string)
```

### 4.19. Modifying Vocabulary <a name="modifying-vocabulary"></a>

TokenMonster provides methods to modify the vocabulary. You can add special tokens, add regular tokens, delete tokens, resize the vocabulary, and enable or disable the UNK token.

**Example:**

```python
# Example of modifying the vocabulary
# Add a special token
tokenizer.modify(add

_special_tokens="[_START_]", add_regular_tokens=None, delete_tokens=None, resize=None, change_unk=None)

# Delete a regular token
tokenizer.modify(add_special_tokens=None, add_regular_tokens=None, delete_tokens=["apple"], resize=None, change_unk=None)
```

### 4.20. Adding Regular Tokens <a name="adding-regular-tokens"></a>

You can add one or more regular tokens to the vocabulary using the `add_token` method.

**Example:**

```python
# Add regular tokens to the vocabulary
tokenizer.add_token(["apple", "banana", "cherry"])
```

### 4.21. Deleting Tokens <a name="deleting-tokens"></a>

To delete one or more regular or special tokens from the vocabulary, use the `delete_token` method.

**Example:**

```python
# Delete tokens from the vocabulary
tokenizer.delete_token(["apple", "[PAD]"])
```

### 4.22. Deleting Tokens by ID <a name="deleting-tokens-by-id"></a>

You can delete one or more regular or special tokens by specifying their token IDs using the `delete_token_by_id` method.

**Example:**

```python
# Delete tokens by ID
tokenizer.delete_token_by_id([42, 0])
```

### 4.23. Adding Special Tokens <a name="adding-special-tokens"></a>

Special tokens play specific roles in tokenization. You can add one or more special tokens to the vocabulary using the `add_special_token` method.

**Example:**

```python
# Add special tokens to the vocabulary
tokenizer.add_special_token(["[_START_]", "[_END_]"])
```

### 4.24. Resizing Vocabulary <a name="resizing-vocabulary"></a>

To change the size of the vocabulary, you can use the `resize` method. This allows you to specify the desired size of the vocabulary.

**Example:**

```python
# Resize the vocabulary to a specific size
tokenizer.resize(10000)
```

### 4.25. Resetting Token IDs <a name="resetting-token-ids"></a>

The `reset_token_ids` method resets the token IDs to be sequential beginning from zero.

**Example:**

```python
# Reset token IDs
tokenizer.reset_token_ids()
```

### 4.26. Enabling UNK Token <a name="enabling-unk-token"></a>

You can enable the UNK (unknown) token in the vocabulary using the `enable_unk_token` method.

**Example:**

```python
# Enable the UNK token
tokenizer.enable_unk_token()
```

### 4.27. Disabling UNK Token <a name="disabling-unk-token"></a>

The `disable_unk_token` method allows you to disable the UNK (unknown) token in the vocabulary.

**Example:**

```python
# Disable the UNK token
tokenizer.disable_unk_token()
```

### 4.28. Disconnecting TokenMonster <a name="disconnecting-tokenmonster"></a>

To gracefully disconnect from the TokenMonster server, use the `disconnect` method.

**Example:**

```python
# Disconnect from TokenMonster server
tokenizer.disconnect()
```

### 4.29. Serializing Tokens <a name="serializing-tokens"></a>

TokenMonster provides the `serialize_tokens` method to serialize tokens from a list of integers or a numpy array into a binary string.

**Example:**

```python
# Serialize tokens
token_ids = [1, 2, 3]
serialized_tokens = tokenizer.serialize_tokens(token_ids)
```

### 4.30. Deserializing Tokens <a name="deserializing-tokens"></a>

You can use the `deserialize_tokens` method to deserialize a binary string into a numpy array of token IDs.

**Example:**

```python
# Deserialize tokens
binary_string = b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00"
deserialized_tokens = tokenizer.deserialize_tokens(binary_string)
```

## 5. Additional Information <a name="additional-information"></a>

### 5.1. Multiprocessing Safety

TokenMonster provides a `load_multiprocess_safe` method that is safe for multiprocessing. When using this method, vocabulary modification is disabled, and tokenization may be slightly slower compared to the regular `load` method.

### 5.2. Supported Character Sets

TokenMonster supports two character sets: "UTF-8" and "None." You can check the character set used by the vocabulary using the `charset` method.

### 5.3. Supported Normalization Options

The vocabulary can have various normalization options applied, including "None," "NFD," "Lowercase," "Accents," "Quotemarks," "Collapse," "Trim," "LeadingSpace," and "UnixLines." You can access the normalization setting using the `normalization` method.

### 5.4. Capcode Levels

The capcode level of the vocabulary can be set to values between 0 and 2 using the `capcode` method. Capcoding is a way to encode multiple tokens using a single token, which can save memory.

### 5.5. Optimization Modes

TokenMonster supports optimization modes from 0 to 5, which affect the memory usage and performance of the library. You can check the optimization mode using the `mode` method.

## 6. Examples <a name="examples"></a>

Let's explore some examples of how to use TokenMonster for tokenization and vocabulary management.

### Example 1: Tokenizing Text

```python
from zeta.tokenizers import TokenMonster

# Initialize TokenMonster with a vocabulary file
tokenizer = TokenMonster("path/to/vocabulary")

# Tokenize a text string
text = "Hello, world!"
token_ids = tokenizer.tokenize(text)
print(token_ids)
```

### Example 2: Decoding Tokens

```python
from zeta.tokenizers import TokenMonster

# Initialize TokenMonster with a vocabulary file
tokenizer = TokenMonster("path/to/vocabulary")

# Decode token IDs into a string
decoded_text = tokenizer.decode([1, 2, 3])
print(decoded_text)
```

### Example 3: Modifying Vocabulary

```python
from zeta.tokenizers import TokenMonster

# Initialize TokenMonster with a vocabulary file
tokenizer = TokenMonster("path/to/vocabulary")

# Add a special token
tokenizer.modify(
    add_special_tokens="[_START_]",
    add_regular_tokens=None,
    delete_tokens=None,
    resize=None,
    change_unk=None,
)

# Delete a regular token
tokenizer.modify(
    add_special_tokens=None,
    add_regular_tokens=None,
    delete_tokens=["apple"],
    resize=None,
    change_unk=None,
)
```

### Example 4: Exporting Vocabulary to YAML

```python
from zeta.tokenizers import TokenMonster

# Initialize TokenMonster with a vocabulary file
tokenizer = TokenMonster("path/to/vocabulary")

# Export the vocabulary to a YAML file
yaml_data = tokenizer.export_yaml()
with open("vocabulary.yaml", "wb") as file:
    file.write(yaml_data)
```

## 7. Conclusion <a name="conclusion"></a>

TokenMonster is a powerful Python module for tokenization and vocabulary management. Whether you're working on natural language processing tasks or need to create custom tokenization pipelines, TokenMonster provides the flexibility and functionality to handle tokenization efficiently. Use the examples and methods provided in this guide to leverage TokenMonster for your projects.
