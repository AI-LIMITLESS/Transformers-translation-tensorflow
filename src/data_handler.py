import os, sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir));
import config
import tensorflow as tf
import tensorflow_text as tf_text
import pathlib

def download_translation_data(filename='fra-eng'):
    """
    Download translation data from google tensorflow storage 

    Parameters
    ----------
    filename : str, optional
        Filename in google storage. Typically, "src-trg" and replacing src and trg with the first 3 digits in corresponding langages. 
        The default is 'fra-eng'.

    Returns
    -------
    pathlib.Path
        Local to textual file in a zip

    """
    path_to_zip = tf.keras.utils.get_file(
        f'{filename}.zip', origin=f'http://storage.googleapis.com/download.tensorflow.org/data/{filename}.zip',
        extract=True)
    path_to_file = pathlib.Path(path_to_zip).parent / f"{filename.split('-')[0]}.txt"
    return path_to_file

def load_translation_data(path: pathlib.Path) -> tuple:
    """
    Reads translation data and return a tuple of two lists of sentences (src & trg) 

    Parameters
    ----------
    path : pathlib.Path
        Path to translation data (text file each sentence in a line and langage separated by '\t' in each row)

    Returns
    -------
    tuple
        A tuple of src and trg list of sentences

    """
    text = path.read_text(encoding='utf-8')

    lines = text.splitlines()
    pairs = [line.split('\t') for line in lines]

    src = [src for _, src in pairs]
    trg = [trg for trg, _ in pairs]

    return src, trg

def create_dataset(src: list, trg: list) -> tf.data.Dataset:
    """
    Create tensorflow dataset object based on lists of sentences

    Parameters
    ----------
    src : list
        List of src sentences (string)
    trg : list
        List of trg sentences (string)

    Returns
    -------
    tf.data.Dataset
        Tensorflow dataset object

    """
    dataset = tf.data.Dataset.from_tensor_slices((src, trg)).shuffle(config.BUFFER_SIZE)
    dataset = dataset.batch(config.BATCH_SIZE)
    return dataset

def tf_lower_and_split_punct(text: str) -> str:
    """
    Preprocesses text input: unicode, lower, filtering chars, separating punct, strip, adding start/end tokens

    Parameters
    ----------
    text : str
        Input text
            Input text to process
    Returns
    -------
    str
        Processed text following the staps described above

    """
    # Split accecented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)
    # Adding [START] and [END] tokens
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

def create_text_vectorizers() -> tuple:
    """
    Create two textvectirizers (including text cleaning) for both src and trg

    Returns
    -------
    tuple
        Tuple of textVectorizers respectively on src and trg

    """
    src_text_processor = tf.keras.layers.TextVectorization(standardize=tf_lower_and_split_punct,
                                                             max_tokens=config.VOCAB_SIZE)
    trg_text_processor = tf.keras.layers.TextVectorization(standardize=tf_lower_and_split_punct,
                                                              max_tokens=config.VOCAB_SIZE)
    return src_text_processor, trg_text_processor