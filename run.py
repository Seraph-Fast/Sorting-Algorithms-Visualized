from tkinter import ttk
from tkinter.ttk import Progressbar
from tkinter import *
from tkinter import messagebox
import SORT as sortlib

w=Tk()


width_of_window = 427
height_of_window = 250
screen_width = w.winfo_screenwidth()
screen_height = w.winfo_screenheight()
x_coordinate = (screen_width/2)-(width_of_window/2)
y_coordinate = (screen_height/2)-(height_of_window/2)
w.geometry("%dx%d+%d+%d" %(width_of_window,height_of_window,x_coordinate,y_coordinate))


w.overrideredirect(1)


s = ttk.Style()
s.theme_use('clam')
s.configure("red.Horizontal.TProgressbar", foreground='red', background='#4f4f4f')
progress=Progressbar(w,style="red.Horizontal.TProgressbar",orient=HORIZONTAL,length=500,mode='determinate',)


def Run1(clicked,q):
    # print("hello")
    algorithm = clicked.get()
    sortlib.Algorithm(algorithm)
    if algorithm == "Select Algorithm":
        messagebox.showerror("Error!", "Please select Algorithm.")

def new_win():
  # w.destroy()
    q=Tk()
    q.title('List Of Sorting Algorithms')
    q.config(bg="black")
    q.geometry('427x250')

    Sorting = [
        'Hybrid Sort',
        'Insertion Sort', 
        'Bubble Sort', 
        'Merge Sort', 
        'Quick Sort', 
        'Heap Sort', 
        'Bucket Sort', 
        'Radix Sort',
        'Count Sort'
        ]
    clicked = StringVar()

    drop = OptionMenu( q , clicked , *Sorting )
    clicked.set("Select Algorithm")
    drop.pack(pady=10)
    drop.config(bg="yellow", activebackground="light green", cursor="hand2")

    NextButton = Button(q, text="Next>", bg="yellow", activebackground="light green",command=lambda:Run1(clicked,q))
    NextButton.pack(pady=20)

    q.mainloop()



def bar():

    l4=Label(w,text='Loading...',fg='white',bg=a)
    lst4=('Calibri (Body)',10)
    l4.config(font=lst4)
    l4.place(x=18,y=210)
    
    import time
    r=0
    for i in range(100):
        progress['value']=r
        w.update_idletasks()
        time.sleep(0.03)
        r=r+1
    
    w.destroy()
    new_win()
        
    
progress.place(x=-10,y=235)

a='#249794'
Frame(w,width=427,height=241,bg=a).place(x=0,y=0)  #249794
b1=Button(w,width=10,height=1,text='Get Started',command=bar,border=0,fg=a,bg='white')
b1.place(x=170,y=200)

######## Label
l1=Label(w,text='Sorting Algorithm Visualizer',fg='white',bg=a)
lst1=('Calibri (Body)',18,'bold')
l1.config(font=lst1)
l1.place(x=50,y=80)

l3=Label(w,text='By Sarim',fg='white',bg=a)
lst3=('Calibri (Body)',13)
l3.config(font=lst3)
l3.place(x=50,y=110)


w.mainloop()


