# Create aliases for tensorflow distributions
import tensorflow as tf

attrs = dir( tf.contrib.distributions ) 
distns = [ attr for attr in attrs if attr[0].isupper() ]

with open('../R/alias.R', 'w') as outscript:
    for distn in distns:
        outscript.write("#' Tensorflow distribution {0}\n#'\n".format(distn))
        outscript.write("#' Alias for tensorflow distribution {0}. See the classes section of \url{{https://www.tensorflow.org/api_docs/python/tf/contrib/distributions}}.\n".format(distn))
        outscript.write("#' @seealso \url{{https://www.tensorflow.org/api_docs/python/tf/contrib/distributions}}\n".format(distn))
        outscript.write("{0} <- tf$contrib$distributions${0}\n\n".format(distn))
