
def validate(query, params):
    return set(params.keys()) <= set(query.keys())

def parseparams(query, params):
    result = {}
    valid = True
    for k,v in params.items():
        ptype = v['type']
        if ptype == 'str':
            result[k] = query[k][0]
        elif ptype == 'int':
            size = v['size'] if 'size' in v else 1
            param = query[k][0].split(',')
            if len(param) != size:
                valid = False
                break
            else:
                try:
                    result[k] = [int(p) for p in param] if size > 1 else int(param[0])
                except:
                    valid = False
                    break
    return valid,result
