import sys
import getopt
    
def analyse_args(args_params):
    short_ids_list = ''
    long_ids_list = []
    usage = 'python {}'.format(sys.argv[0])
    args_values = {}
    for short_id, long_id, _, default_value in args_params:
        short_ids_list += short_id + ':'
        long_ids_list.append(long_id + '=')
        usage += ' -{} [{}]'.format(short_id, long_id)
        args_values[long_id] = default_value
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], short_ids_list, long_ids_list)
    except getopt.GetoptError:
        print(usage) ; sys.exit(2)
    
    for opt, arg in opts:
        recognized = False
        for short_id, long_id, process_arg_func, _ in args_params:
            if opt in ('-{}'.format(short_id), '--{}'.format(long_id)):
                args_values[long_id] = process_arg_func(arg)
                recognized = True
                break
        if not recognized:
            print(usage) ; sys.exit(2)
    
    return args_values
