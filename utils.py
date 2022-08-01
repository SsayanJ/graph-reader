def is_number(s):
    """ Returns True is string is a number. """
    return s.replace('.','',1).replace('-', '', 1).isdigit()
