import serial
serialPort = serial.Serial()
# 已修改
class dh_device(object) :

    def connect_device(self,portname, Baudrate) :
        ret = -1
        #print('portname: ', portname)
        serialPort.port = portname
        serialPort.baudrate = Baudrate
        serialPort.bytesize = 8
        serialPort.parity = 'N'
        serialPort.stopbits = 1
        serialPort.set_output_flow_control = 'N'
        serialPort.set_input_flow_control = 'N'
        serialPort.timeout = 1.0    # 1s超时

        serialPort.open()
        if(serialPort.isOpen()) :
            print('Serial Open Success')
            ret = 0
        else :
            print('Serial Open Error')
            ret = -1
        return ret

    def disconnect_device(self):
        if(serialPort.isOpen()) :
            serialPort.close()
        else :
            return

    def device_wrire(self, write_data) :
        if(serialPort.isOpen()) :
            write_lenght = serialPort.write(write_data)
            if(write_lenght == len(write_data)) :
                return write_lenght
            else :
                print('write error ! send_buff :',write_data)
                return 0
        else :
            return -1

    # def device_read(self, wlen) :
    #     # responseData = [0,0,0,0,0,0,0,0]
    #     if(serialPort.isOpen()) :
    #         # responseData = serialPort.readline(wlen)    # 会一直读取直到遇到换行符 \n 或达到 wlen 字节。Modbus RTU 协议是二进制数据，不保证包含换行符，所以 readline 很可能永远读不到换行符，导致永久阻塞。
    #         # 关键修改：使用 read 而不是 readline
    #         responseData = self.serialPort.read(wlen)
    #         #print('read_buff: ',responseData.hex())
    #         return responseData
    #     else :
    #         return -1

    def device_read(self, wlen):    # 已修改
        if not serialPort.isOpen():
            return b''
        response_data = serialPort.read(wlen)
        return response_data

    """description of class"""


