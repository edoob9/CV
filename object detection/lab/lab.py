
import csv
import re
from enum import Enum

#'filename', 'file_size', 'file_attributes', 'region_count', 'region_id', 'region_shape_attributes', 'region_attributes'

class ColumnInfo(Enum):
    # 해당 페이지에서 저장한 csv파일을 보면 이렇게 6개의 속성으로 이뤄져있기때문에 지정해준 것이다.
    filename = 0 
    file_size= 1
    file_attributes = 2   
    region_count = 3  
    region_id = 4 
    region_shape_attributes = 5 
    region_attributes = 6


def read_class_names(class_file_name): #class_file_name : lion, elephant - class 2
    names = {}
    class_file_name = 'classnames.txt'
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[name.strip('\n')] = ID 
    return names


class AnnotationFileMaker:

    origin_data=[]
    parsedData={}
    return_str=''
 
    def __init__(self,filepath):
        with open(filepath, 'r',encoding='utf-8') as csvfile:
            rdr = csv.reader(csvfile)
            for line in rdr:
                self.origin_data.append(line)             
        self.origin_data=self.origin_data[1:]
        self.parser(self.origin_data, class_file_name)
        self.put_text(self.parsedData)
        self.return_Data()

    def parser(self,data):
        prev=''
        for i in range(len(data)):
            if data[i][ColumnInfo.filename.value] !=prev:
                self.parsedData[data[i][ColumnInfo.filename.value]]=[]
            coordinates = data[i][ColumnInfo.region_shape_attributes.value]
        
            temp = re.findall(r'\d+', coordinates)
            coordinates = list(map( str, temp))
            obj_idx = data[i][ColumnInfo.region_attributes.value].split(':')[1][:-1]
            obj_idx = int(obj_idx[1:-1])
            coordinates.extend(str(obj_idx))
            coordinates = ",".join(coordinates)
            self.parsedData[data[i][ColumnInfo.filename.value]].append([coordinates])
            prev=data[i][ColumnInfo.filename.value]

    def put_text(self,parsedData):
        for item in parsedData:
            self.return_str += item
            self.return_str += ' '
            temp=''
            for i in range(len(parsedData[item])):            
                temp += parsedData[item][i][0]
                temp += ','
            self.return_str += temp[:-1]
            self.return_str+='\n'

    def return_Data(self):
        return self.return_str

def main():
    filepath = 'label_test.csv'

    data = AnnotationFileMaker(filepath).return_Data()
    print(data)

    with open('./result.txt', 'w', encoding='utf-8') as f:
        # write a row to the csv file
        f.write(data)


if __name__ == '__main__':
    main()
