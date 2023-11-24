import numpy as np
import tokenmonster


class TokenMonster:
    """
    A class that encapsulates the functionality of the tokenmonster library.

    >>> from zeta.tokenizers import TokenMonster
    >>> tokenizer = TokenMonster("englishcode-32000-consistent-v1")
    >>> tokenizer.tokenize("Hello world!")
    """

    def __init__(self, path):
        """
        Initializes the TokenMonster class and loads a vocabulary.

        Args:
            path (str): A filepath, URL or pre-built vocabulary name.
        """
        self.vocab = tokenmonster.load(path)

    def set_local_directory(self, dir=None):
        """
        Sets the local directory for TokenMonster.

        Args:
            dir (str, optional): The local directory to use. Defaults to None.
        """
        tokenmonster.set_local_directory(dir)

    def load(self, path):
        """
        Loads a TokenMonster vocabulary from file, URL or by name.

        Args:
            path (str): A filepath, URL or pre-built vocabulary name.
        """
        self.vocab = tokenmonster.load(path)

    def load_multiprocess_safe(self, path):
        """
        Loads a TokenMonster vocabulary from file, URL or by name. It's safe for multiprocessing,
        but vocabulary modification is disabled and tokenization is slightly slower.

        Args:
            path (str): A filepath, URL or pre-built vocabulary name.
        """
        self.vocab = tokenmonster.load_multiprocess_safe(path)

    def new(self, yaml):
        """
        Creates a new vocabulary from a YAML string.

        Args:
            yaml (str): The YAML file.
        """
        self.vocab = tokenmonster.new(yaml)

    def save(self, fname):
        """
        Saves the current vocabulary to a file.

        Args:
            fname (str): The filename to save the vocabulary to.
        """
        self.vocab.save(fname)

    def export_yaml(self, order_by_score=False):
        """
        Exports the vocabulary as a YAML file, which is returned as a bytes string.

        Args:
            order_by_score (bool, optional): If true the tokens are order by score instead of alphabetically. Defaults to False.

        Returns:
            bytes: The vocabulary in YAML format.
        """
        return self.vocab.export_yaml(order_by_score)

    def tokenize(self, text):
        """
        Tokenizes a string into tokens according to the vocabulary.

        Args:
            text (str): A string or bytes string, or list of strings or bytes strings.

        Returns:
            numpy array: The tokens IDs
        """
        return self.vocab.tokenize(text)

    def tokenize_count(self, text):
        """
        Same as tokenize, but it returns only the number of tokens.

        Args:
            text (str): A string or bytes string, or list of strings or bytes strings.

        Returns:
            int: The number of tokens for each input string
        """
        return self.vocab.tokenize_count(text)

    def decode(self, tokens):
        """
        Decodes tokens into a string.

        Args:
            tokens (int, list of int, or numpy array): The tokens to decode into a string.

        Returns:
            str: The composed string from the input tokens.
        """
        return self.vocab.decode(tokens)

    def decoder(self):
        """
        Returns a new decoder instance used for decoding tokens into text.

        Returns:
            tokenmonster.DecoderInstance: A new decoder instance.
        """
        return self.vocab.decoder()

    def get_dictionary(self):
        """
        Returns a dictionary of all tokens in the vocabulary.

        Returns:
            list: A list of dictionaries where the index is the token ID and each is a dictionary.
        """
        return self.vocab.get_dictionary()

    def charset(self):
        """
        Returns the character set used by the vocabulary.

        Returns:
            str: The character set used by the vocabulary. Possible values are "UTF-8", "None".
        """
        return self.vocab.charset()

    def normalization(self):
        """
        Returns the normalization of the vocabulary.

        Returns:
            str: The normalization of the vocabulary. Possible values are "None", "NFD", "Lowercase", "Accents", "Quotemarks", "Collapse", "Trim", "LeadingSpace", "UnixLines".
        """
        return self.vocab.normalization()

    def capcode(self):
        """
        Returns the capcode level of the vocabulary.

        Returns:
            int: The capcode level (0-2).
        """
        return self.vocab.capcode()

    def mode(self):
        """
        Returns the optimization mode of the vocabulary.

        Returns:
            int: The optimization mode (0-5).
        """
        return self.vocab.mode()

    def id_to_token(self, id):
        """
        Get the token string from a single token ID, in its capcode-encoded form.

        Args:
            id (int): The token ID.

        Returns:
            str or None: The token string corresponding to the input ID. None if the ID is not in the vocabulary.
        """
        return self.vocab.id_to_token(id)

    def id_to_token_decoded(self, id):
        """
        Get the token string from a single token ID, in its capcode-decoded form.

        Args:
            id (int): The token ID.

        Returns:
            str or None: The token string corresponding to the input ID. None if the ID is not in the vocabulary.
        """
        return self.vocab.id_to_token_decoded(id)

    def token_to_id(self, token):
        """
        Returns the ID of a single token.

        Args:
            token (str): The token to get the ID for.

        Returns:
            int or None: The ID of the token. None if the token is not in the vocabulary.
        """
        return self.vocab.token_to_id(token)

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
            delete_tokens (str or list of str, optional): Regular or Special tokens to delete.
            resize (int, optional): Resizes the vocabulary to this size.
            change_unk (bool, optional): If set, it enables or disables the UNK token.

        Returns:
            int: The new size of the vocabulary.
        """
        return self.vocab.modify(
            add_special_tokens,
            add_regular_tokens,
            delete_tokens,
            resize,
            change_unk,
        )

    def add_token(self, token):
        """
        Add one or more regular tokens.

        Args:
            token (str or list of str): The regular tokens to add.

        Returns:
            int: The new size of the vocabulary.
        """
        return self.vocab.add_token(token)

    def delete_token(self, token):
        """
        Delete one or more regular or special tokens.

        Args:
            token (str or list of str): The tokens to delete.

        Returns:
            int: The new size of the vocabulary.
        """
        return self.vocab.delete_token(token)

    def delete_token_by_id(self, id):
        """
        Delete one or more regular or special token by specifying the token ID.

        Args:
            id (int or list of int): The IDs of the tokens to delete.

        Returns:
            int: The new size of the vocabulary.
        """
        return self.vocab.delete_token_by_id(id)

    def add_special_token(self, token):
        """
        Add one or more special tokens.

        Args:
            token (str or list of str): The special tokens to add.

        Returns:
            int: The new size of the vocabulary.
        """
        return self.vocab.add_special_token(token)

    def resize(self, size):
        """
        Changes the size of the vocabulary.

        Args:
            size (int): The new size of the vocabulary.

        Returns:
            int: The new size of the vocabulary.
        """
        return self.vocab.resize(size)

    def reset_token_ids(self):
        """
        Resets the token IDs to be sequential beginning from zero.
        """
        self.vocab.reset_token_ids()

    def enable_unk_token(self):
        """
        Enables the UNK token.

        Returns:
            int: The new size of the vocabulary.
        """
        return self.vocab.enable_unk_token()

    def disable_unk_token(self):
        """
        Disables the UNK token.

        Returns:
            int: The new size of the vocabulary.
        """
        return self.vocab.disable_unk_token()

    def disconnect(self):
        """
        Disconnects and closes tokenmonsterserver.
        """
        tokenmonster.disconnect()

    def serialize_tokens(self, integer_list):
        """
        Serializes tokens from a list of ints or numpy array into a binary string.

        Args:
            integer_list (list of int or numpy array): The tokens to serialize.

        Returns:
            bytes: The serialized binary string.
        """
        return self.vocab.serialize_tokens(integer_list)

    def deserialize_tokens(self, binary_string):
        """
        Deserializes a binary string into a numpy array of token IDs.

        Args:
            binary_string (bytes): The binary string to deserialize.

        Returns:
            np.array: The deserialized tokens.
        """
        return self.vocab.deserialize_tokens(binary_string)
