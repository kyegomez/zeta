import os
import warnings
import logging

# Immediately suppress warnings
warnings.filterwarnings("ignore")

# Set environment variables to minimize logging before importing any modules
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

# Force NumExpr to use minimal threads to reduce its logging output
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

def disable_warnings_and_logs():
    # Attempt to reduce TensorFlow verbosity if installed
    try:
        import tensorflow as tf
        tf.get_logger().setLevel(logging.ERROR)
        tf.autograph.set_verbosity(3)
    except ImportError:
        pass

    # Reduce logging for known verbose libraries
    logging.getLogger().setLevel(logging.CRITICAL)  # Suppress most logs globally
    
    # Suppress specific verbose loggers known to output unwanted messages
    for logger_name in ['transformers', 'torch', 'tensorflow', 'numexpr']:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    # Specifically target the NumExpr logger if it's being stubborn
    logging.getLogger('numexpr').setLevel(logging.CRITICAL)

# Run the suppression function at the start
disable_warnings_and_logs()

# Ensure to place any of your script's import statements here, after the call to disable_warnings_and_logs()
