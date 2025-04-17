class FileNumberProcessor:
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def process_single_line(self, line, line_number):  
        line = line.strip()
        try:
            number = float(line)
            result = round(number ** 2)
            print(f"成功处理数字:{line},平方:{result}")
        except ValueError as e:
            print(f"无效数据:第{line_number}行内容'{line}'不能转换为数字")

    def process_file(self, file_path):
        print(f"开始处理文件:{file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                line_number = 0
                for line in file:
                    line_number += 1
                    self.process_single_line(line, line_number)
                print(f"成功处理文件:{file_path}")
        except FileNotFoundError:
            print(f"文件{file_path}不存在")
        except PermissionError:
            print(f"无权限读取{file_path}")
        except Exception as e:
            print(f"处理文件{file_path}时发生未知错误: {e}")

    def run(self):
        for file_path in self.file_paths:
            self.process_file(file_path)

if __name__ == "__main__":
    file_paths = ['第7周\\numbers.txt', 'another.txt']
    processor = FileNumberProcessor(file_paths)
    processor.run()