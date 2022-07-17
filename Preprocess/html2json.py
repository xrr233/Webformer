import re
import json
from tqdm import tqdm
from bs4 import BeautifulSoup
import bs4
import os
import sys
sys.setrecursionlimit(10000)
class HTMLCleaner(object):
    def __init__(self,input_file,output_file):
        self.input_file = input_file
        self.output_file = output_file
        
    def clean_tag(self,x):
        x = x.group().split()
        if x[0]=='<':
            y = x[0]+x[1].strip('>')+'>'
        else:
            y = x[0].strip('>')+'>'
        return y
    
    def clean_html(self):
        style_regex = "(?:<style.*?>(?:.|[\r\n])*?</style>|<script.*?>(?:.|[\r\n])*?</script>)"
        all_tag_regex = "(?:<(?:!|/?[a-zA-Z]+).*?/?>)"
        close_tag_regex = '(<(?:!|/?[a-zA-Z]+)[^>]*?/>){1}?'
        annotation_regex = '(?:<!--(?:.|[\r\n])*?-->)'
        close_tag_pattern = re.compile(close_tag_regex)
        annotation_pattern = re.compile(annotation_regex)
        all_tag_pattern = re.compile(all_tag_regex)
        style_pattern = re.compile(style_regex)
        tags = []
        annotation = []
        close_tag = []
        all_tag = []
        style = []
        with open(self.input_file,'r')as f:
            line = f.read()
            annotation.extend(annotation_pattern.findall(line))
            line = re.sub(annotation_regex,'',line)
            close_tag.extend(close_tag_pattern.findall(line))
            line = re.sub(close_tag_regex,'',line)
            style.extend(style_pattern.findall(line))
            line = re.sub(style_regex,'',line)
            all_tag.extend(all_tag_pattern.findall(line))
            line = re.sub(all_tag_regex,self.clean_tag,line)
        self.line = line
    
    def if_text(self,node):
        if node.get_text('|',strip=True) == "":
            return 0
        else:return 1
    
    
    def get_tree(self,root):
        delete_list = []
        for child in root.children:
            if type(child) == bs4.element.Tag:
                if not self.if_text(child):
                    delete_list.append(child)
                else:
                    child = self.get_tree(child)
            elif type(child) == bs4.element.NavigableString:
                if str(child).strip() == "":
                    delete_list.append(child)
                
        for item in delete_list:
            item.extract()
        return root
    
    def merge_tree(self,root,k):
        i = 0
        while(i<len(root.contents)):
            if(type(root.contents[i])==bs4.element.Tag):
                if len(root.contents[i].contents)<=k:
                    tmp = root.contents[i]
                    j = i
                    del root.contents[i]
                    for item in tmp.contents:
                        root.contents.insert(j,item)
                        j+=1
                else:
                    root.contents[i] = self.merge_tree(root.contents[i],k)
                    i+=1
            else:
                i+=1
        return root

                    
    def clean_tree(self):
        root = BeautifulSoup(self.line,'html.parser').html
        root = self.get_tree(root)
        root = self.merge_tree(root,2)
        self.root = root

    def store(self):
        with open(self.output_file,'w+')as f:
            f.write(str(self.root))

class HTMLStorer(object):
    def __init__(self,input_file=None,html=None):
        if not input_file and not html:
            raise ValueError("lack of input file or html")
        if input_file:
            with open(input_file,'r')as f:
                self.html_text = f.read()
        else:
            self.html_text = html
        self.soup = BeautifulSoup(self.html_text,'html.parser')
        self.root = self.soup
        self.root.depth = 0 
        self.depth = {}
        self.idx = 0
        self.data = []
        self.root = self.add_text_node(self.root)
        self.get_index(self.root)
        self.get_data(self.root)
    
    def add_text_node(self,root):
        for child in root.children:
            try:
                if type(child)== bs4.element.Tag and child.name != "textnode":
                    child = self.add_text_node(child)
                elif type(child)== bs4.element.NavigableString:
                    child.wrap(self.soup.new_tag("textnode"))
            except:
                print(root)
        return root
        
    def get_tag_text(self,node):
        line = ""
        for child in node.children:
            if type(child) == bs4.element.NavigableString:
                x = str(child).strip().replace('\n',' ')
                if x != "":
                    line = line+'\t'+x.strip()
        return line.strip()
    
    def get_index(self,node):
        for child in node.children:
            if type(child) != bs4.element.Tag:
                continue
            child.idx = self.idx
            child.depth = node.depth+1
            self.depth[child.depth] = self.depth[child.depth]+1 if child.depth in self.depth else 1
            self.idx+=1
            self.get_index(child)
    
    def get_data(self,node):
        for child_node in node.children:
            if type(child_node)!= bs4.element.Tag:
                continue
            else:
                name = child_node.name
                node_id = child_node.idx
                node_text = self.get_tag_text(child_node)
                node_child_idx = []
                for item in child_node.children:
                    if type(item) == bs4.element.Tag:
                        node_child_idx.append(item.idx)
                line = {"name":name,"id":node_id,"text":node_text,"children":node_child_idx}
                self.data.append(line)
                self.get_data(child_node)
    
    def store(self,output_file):
        with open(output_file,'a')as g:
            g.write(str(self.idx)+'\t'+json.dumps(self.depth,ensure_ascii=False)+'\n')


dir_path = './data/endata/'
new_dir_path = './data/endata_new_clean/'
output_path = './data/wiki_html_all.json'
if __name__ == "__main__":
    with open(output_path,'w')as g:
        for root, dirs, files in os.walk(dir_path):
            for dir in tqdm(dirs):
                sub_dir_path = os.path.join(root,dir)
                for sub_root,sub_dirs,sub_files in os.walk(sub_dir_path):
                    for f in tqdm(sub_files):
                        #index = f.split('.')[0]
                        path = os.path.join(sub_root,f)
                        clean_path = new_dir_path+'_clean.html'
                        cleaner = HTMLCleaner(path,clean_path)
                        cleaner.clean_html()
                        #cleaner.clean_tree()
                        storer = HTMLStorer(input_file=None,html=cleaner.line)
                        g.write(json.dumps(storer.data,ensure_ascii=False)+'\n')
