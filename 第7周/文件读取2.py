class FileNumberProcessor:
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def process_single_line(self, line, line_number):  
        # 处理单行
        con = line.strip()#去除首尾空白字符
        if con:
        #如果该行不空 转为float
            try:
                con = float(con)
                print(f'成功处理数字:{round(con)},平方:{round(con * con)}')
            except ValueError:
                print(f"无效数据:第{line_number}行内容'{con}'不能转换为数字")

    def process_file(self, file_path):
        try:
            print(f'开始处理文件:{file_path}')
            with open(file_path,'r')as f:
                for line_num, line in enumerate(f ,start=1):
                        self.process_single_line(line,line_num)
            print(f'成功处理文件:{file_path}')
        except FileNotFoundError:
            print(f'文件{file_path}不存在')
        except PermissionError:
            print(f'无权限读取{file_path}')
        except Exception as e:
            print(f'处理文件{file_path}时发生未知错误:{e}')


    def run(self):
        for file_path in self.file_paths:
            self.process_file(file_path)

if __name__ == "__main__":
    file_paths = ['第7周\\numbers.txt', 'another.txt']
    processor = FileNumberProcessor(file_paths)
    processor.run()