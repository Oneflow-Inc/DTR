#coding=UTF-8
import time
import oneflow.experimental as flow

tensor_info_dict = {}#这是一个字典，键：tensor的位置，即第几个tensor，值：0/内存大小(是否在内存中)，tensor进入显存的时间，产生该tensor所花费的时间，计算历史(op,源tensor)

     
def get_size(res):
    size=1              
    for x in res.shape:
        size *= x
    assert(res.dtype==flow.float32)
    return size*4
        

def get_available_mem(gpu_id):
    #获取当前CPU的内存信息
    mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
    mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
    '''计算总占用内存
    for x in tensor_info_dict:
        mem_used+=x[1][0]
    '''
    return mem_free

    
    
def search_tensor_to_release(tensor_info_dict):
    for x in tensor_info_dict:
        min_tensor_value=float('inf')
        min_tensor=None
        stay_mem_time=time.time()-tensor_info_dict[x][2]
        temp_value=tensor_info_dict[x][2]/(stay_mem_time*tensor_info_dict[x][0])#估价函数，算子的运行时间/（待在显存的时长*占用的内存）
        if(temp_value<min_tensor_value):
            min_tensor_value=temp_value
            min_tensor=x
    return min_tensor
        
    
def chongjisuan(input_tensor):
    #读取x的计算历史：op，inputs
    f=tensor_info_dict[input_tensor][3][0]
    #计算calculation_time
    start_time = time.time()
    res=f(tensor_info_dict[input_tensor][3][1],tensor_info_dict[input_tensor][3][1])
    end_time = time.time()
    calculation_time = end_time - start_time
    #计算内存大小
    size=get_size(res) 
    #存入显存
    tensor_info_dict[input_tensor]=[size, time.time(), calculation_time, (f, tensor_info_dict[input_tensor][3][1],tensor_info_dict[input_tensor][3][1])]
    return 
    
    

def rem(f):
    #把一些相关的tensor设置成不可丢弃
    
    def new_op(*args, **kwargs):                                            #args和kwargs是可变参数
        for input_tensor in args:
            if not isinstance(input_tensor, flow.Tensor):
                continue
            if tensor_info_dict[input_tensor][0] == 0:
                chongjisuan(input_tensor)                                   #这个重计算函数chongjisuan()要写
        
        #进行到这说明用到的tensor在内存中
        start_time = time.time()

        res = f(*args, **kwargs)                                            #这个执行函数

        end_time = time.time()
        calculation_time = end_time - start_time                            #得到执行时间

        size = get_size(res)                                                #得到执行结果的大小get_size()要写
        tensor_info_dict[res] = [size, time.time(), calculation_time, (f, args, kwargs)]    #1是说存在在内存中，(f, args, kwargs)记录这个tensor是怎么得到的

        if size > get_available_mem():                                      #get_avaiable_mem()函数需要写
            x = search_tensor_to_release(tensor_info_dict)                  #search_tensor_to_release()需要写，返回搜索到的tensor
            tensor_info_dict[x][0] = 0
            tensor_info_dict[x][1] = 0
        return res

    return new_op

def main():
    #x0-(x0*2+x0)
    start_time=time.time()
    x0 = flow.Tensor(1, 3, 224, 224)
    end_time=time.time()
    calculation_time=end_time-start_time
    tensor_info_dict[0]={'0':[get_size(x0),time.time(),calculation_time,('source',0,0)]}
    '''test'''
    print(x0)
    for x in tensor_info_dict:
        print(tensor_info_dict[x])
    '''test'''

    x1 = rem(flow.mul)(x0, 2)
    for x in tensor_info_dict:
        print(tensor_info_dict[x])
    x2 = rem(flow.add)(x0, x1)  #这种情况下，没法释放，x0,x1,x2
    for x in tensor_info_dict:
        print(tensor_info_dict[x])
    x3 = rem(flow.sub)(x0, x2)  #释放x1，加上前面要把x0释放掉！
    for x in tensor_info_dict:
        print(tensor_info_dict[x])

if __name__=='__main__':
    main()
