#!/usr/bin/env python
# nltk_download.py

# BEFORE running this script, and ensure that you have downloaded
# the requirements in requirements.txt.

import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('wordnet')