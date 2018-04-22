import re

from porterStemmer import porterStemmer
from getVocabList import getVocabList


def regexprep(string, pattern, repl):
    rx = re.compile(pattern)
    contents = rx.sub(repl, string)
    return contents


def processEmail(email_contents):
    """preprocesses a the body of an email and
    returns a list of word_indices
    word_indices = PROCESSEMAIL(email_contents) preprocesses
    the body of an email and returns a list of indices of the
    words contained in the email.
    """

    # Load Vocabulary
    vocabList = getVocabList()

    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================

    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers

    # hdrstart = strfind(email_contents, ([chr(10) chr(10)]))
    # email_contents = email_contents(hdrstart(1):end)

    # Lower case
    email_contents = str.lower(email_contents)

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = regexprep(email_contents, '<[^<>]+>|\n', ' ')

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = regexprep(email_contents, '[0-9]+', 'number ')

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = regexprep(email_contents, '(http|https)://[^\s]*',
                               'httpaddr ')

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr ')

    # Handle $ sign
    email_contents = regexprep(email_contents, '[$]+', 'dollar ')

    # ========================== Tokenize Email ===========================

    # Output the email to screen as well
    print('\n==== Processed Email ====\n\n', end='')

    # Process file
    l = 0

    # Remove any non alphanumeric characters
    email_contents = regexprep(email_contents, '[^a-zA-Z0-9 ]', '').split()

    for word in email_contents:

        # Tokenize and also get rid of any punctuation
        # str = re.split('[' + re.escape(' @$/#.-:&*+=[]?!(){},''">_<#')
        #                                + chr(10) + chr(13) + ']', str)

        # Stem the word
        # (the porterStemmer sometimes has issues, so we use a try catch block)
        try:
            word = porterStemmer(str.strip(word))
        except Exception as ex:
            print(ex)
            word = ''
            continue

        # Skip the word if it is too short
        if len(word) < 1:
            continue

        # Look up the word in the dictionary and add to word_indices if
        # found
        # ====================== YOUR CODE HERE ======================
        # Instructions: Fill in this function to add the index of str to
        #               word_indices if it is in the vocabulary. At this point
        #               of the code, you have a stemmed word from the email in
        #               the variable str. You should look up str in the
        #               vocabulary list (vocabList). If a match exists, you
        #               should add the index of the word to the word_indices
        #               vector. Concretely, if str = 'action', then you should
        #               look up the vocabulary list to find where in vocabList
        #               'action' appears. For example, if vocabList{18} =
        #               'action', then, you should add 18 to the word_indices
        #               vector (e.g., word_indices = [word_indices  18] ).
        #
        # Note: vocabList{idx} returns a the word with index idx in the
        #       vocabulary list.
        #
        # Note: You can use strcmp(str1, str2) to compare two strings (str1 and
        #       str2). It will return 1 only if the two strings are equivalent.
        #
        for key in vocabList:
            if word == vocabList[key]:
                word_indices.append(key)
        # =============================================================

        # Print to screen, ensuring that the output lines are not too long
        if l + len(word) + 1 > 78:
            print('\n', end='')
            l = 0

        print('%s ' % word, end='')
        l = l + len(word) + 1

    # Print footer
    print('\n=========================\n', end='')

    return word_indices
