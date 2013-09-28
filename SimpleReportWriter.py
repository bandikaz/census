import matplotlib.pyplot as plt
import cgi,os,shutil


# (very) simple class to generate html reports
# we can create tables, titles, text elements and insert pyplots

class SimpleReportWriter:
    
    def title(self,txt,level=1):
        self._closeTable()
        if level==1:
            self._write('<hr />')
        self._write('<h%d>%s</h%d>\n'%(level,cgi.escape(str(txt)),level))
        
    def table(self,columns):
        self._write('<table class="sortable">\n<tr class="header">\n')
        for c in columns :
            self._write('<th>%s</th>'%cgi.escape(str(c)))
        self._write('</tr>\n')
        self.insideTable = True
        
    def row(self,values):
        assert(self.insideTable)
        self._write('<tr>')
        for c in values :
            self._write('<td>%s</td>'%cgi.escape(str(c)))
        self._write('</tr>\n')
            
    def text(self,txt):
        self._closeTable()
        self._write('<p>%s</p>\n'%cgi.escape(str(txt)))
        
    def pre(self,txt):
        self._closeTable()
        self._write('<pre>%s</pre>\n'%cgi.escape(str(txt)))
        
    # insert the current plot
    def plot(self):
        self._closeTable()
        image_file = 'fig%d.png'%(self.fig_count)
        plt.tick_params(axis='both', labelsize=7)
        plt.xlabel("...",labelpad=35)
        plt.tight_layout()
        plt.savefig(self.dir+'/'+image_file,dpi=80)
        
        self.fig_count+=1
        self._write('<img src="%s">\n'%image_file)
        
    def close(self):
        self._closeTable()
        self._put_footer()
        self.fp.close()
    
    def __init__(self,html_output_dir):
        try:
            os.makedirs(html_output_dir)
        except:
            pass # bad practice in general but here we don't really care 
            
        self.fp = open(html_output_dir+'/index.html','w')
        self.dir = html_output_dir
        self.insideTable = False
        self.fig_count = 0
        self._put_header()

        
        shutil.copy('sorttable.js',html_output_dir)
        
    def _write(self, txt):
        self.fp.write(txt)
        self.fp.flush()
        
    def _closeTable(self):
        if self.insideTable:
            self.insideTable = False
            self._write('</table>\n')
    
    def _put_header(self):
        self._write('''<html>
        <head>
            <style>body { font:11px arial,sans-serif; } 
            table { border: 1px solid #AAAAAA; margin:10px; }  
            tr.header { border-bottom:1px solid #AAAAAA; background-color:#AAAAAA; font-weight:bold; } 
            td { padding:5px; border-right:1px solid #AAAAAA; }
            th { border-right:1px solid #AAAAAA; }
            </style>
            
            <script src="sorttable.js"></script>
        </head>
        <body>
        \n''')
        
    def _put_footer(self):
        self._write('</body></html>')