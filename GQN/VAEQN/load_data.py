# Returns data from specified model or batch number

def get_data(test_loader, randn):
    for t, data in enumerate(test_loader, 0):
        if t == randn:
            data_tmp = data
            return data_tmp