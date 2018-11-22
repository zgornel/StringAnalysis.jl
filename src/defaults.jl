# Defaults used throughout the package. Their role is to provide default values
# for various arguments and keyword arguments and a single point of control
# for the default behaviour of package functions.
#
# Note: Constants that have local scope (such as preprocessing.jl constants)
#       should not be added here.

# Regular expression on which to split text into tokens;
# It used by the tokenize_fast function
const DEFAULT_TOKENIZATION_REGEX = r"(,|\n|\r|\:|\\|\/|;|\.|\[|\]|\{|\}|\'|\`|\"|\"|\?|\!|\=|\~|\&|\s+)"
