
def plot_check_system(test_name, description, plot_fun, args, kwargs):

    print('\n'*2)
    print('--------  {}  --------'.format(test_name))
    print('# Testing {}'.format(plot_fun.__name__))
    print('-------- Test description: -----------')
    print(description)
    print('--------------------------------------')
    plot_fun(*args, **kwargs)
    print('# Function executed ')
    ans = raw_input('Did the plot produce the expected results? [y / (default)n]: ')
    if ans.lower() == 'y':
        return True
    elif ans.lower() == 'n' or len(ans) == 0:
        return False
    else:
        plot_check_system(test_name, description, plot_fun, args, kwargs)
    print('\n'*2)


