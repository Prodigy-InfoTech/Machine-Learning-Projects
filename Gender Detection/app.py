import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog as fd
from main import genderDetect
import urllib.request


def detectGender():
    if(url_entry.get().strip(" ")!=""):
        urllib.request.urlretrieve(url_entry.get(),"download.png")
        genderDetect("download.png")
        url_entry.delete(0,'end')
    else:
        messagebox.showinfo("Result","Please enter image url")
def selectImage():
    path=fd.askopenfilename()
    genderDetect(path)

window=tk.Tk()
window.title("Gender Detection")
window.geometry("400x500")
window.configure(bg="#000aff")
window.resizable(0,0)

title_label=tk.Label(window,text="Gender Detection",font=("Arial",20,"bold"),bg="#000AFF",fg="#fff")
title_label.pack(pady=20)

url_entry=tk.Entry(window,font=("Arial",15,""),width=350)
url_entry.pack(ipadx=10,ipady=2,padx=15)

detect_btn=tk.Button(window,text="Detect",font=("Arial",10,"bold"),command=detectGender)
detect_btn.configure(bg="#fff",fg="#000aff")
detect_btn.pack(pady=20,ipadx=20,ipady=3)

divider=tk.Label(window,text="( or )",font=("Arial",13,"bold"),bg="#000AFF",fg="#fff")
divider.pack(pady=20)

select_btn=tk.Button(window,text="Select an image",font=("Arial",10,"bold"),command=selectImage)
select_btn.configure(bg="#fff",fg="#000aff")
select_btn.pack(pady=10,ipadx=10,ipady=2)


window.mainloop()
