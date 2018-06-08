from tkinter import *
from PIL import ImageTk, Image

class Ethnicity_Labeler:

    def __init__(self, master):
        self.master = master
        master.title("Ethnicity Labeler")
        self.THRESHOLD = 0.35
		
        with open('still_uncertain.txt') as f:
            lines = [line for line in f]
        self.names = [line.split(',')[0] for line in lines]
        self.confidences = [float(line.split(',')[1]) for line in lines]

        self.outf = open("uncertain_labeled_" + str(self.THRESHOLD) + ".txt", "w+")

        self.canvas = Canvas(root, width = 250, height = 250)
        self.canvas.pack()
		
        self.label_text = StringVar()
        self.label_text.set("Ethnicity Labeler for imdb_wiki dataset")
        self.label = Label(master, textvariable=self.label_text)
        self.label.pack()

        self.index = 0
        display(self)

        self.greet_button = Button(master, text="caucasian", command=self.caucasian)
        self.greet_button.pack()
        self.greet_button = Button(master, text="black", command=self.black)
        self.greet_button.pack()
        self.greet_button = Button(master, text="indian", command=self.indian)
        self.greet_button.pack()
        self.greet_button = Button(master, text="asian", command=self.asian)
        self.greet_button.pack()
        self.greet_button = Button(master, text="others", command=self.others)
        self.greet_button.pack()

        self.close_button = Button(master, text="Close", command=self.quit)
        self.close_button.pack()

    def greet(self):
        print("Greetings!")

    def caucasian(self):
        self.label_text.set("Choosing caucasian!")
        self.outf.write(self.names[self.index] + ",caucasian\n")
        display(self)
	
    def black(self):
        self.label_text.set("Choosing black!")
        self.outf.write(self.names[self.index] + ",black\n")
        display(self)

    def indian(self):
        self.label_text.set("Choosing indian!")
        self.outf.write(self.names[self.index] + ",indian\n")
        display(self)

    def asian(self):
        self.label_text.set("Choosing asian!")
        self.outf.write(self.names[self.index] + ",asian\n")
        display(self)

    def others(self):
        self.label_text.set("Choosing others!")
        self.outf.write(self.names[self.index] + ",others\n")
        display(self)

    def quit(self):
        self.outf.close()
        self.master.quit()
		
def display(self):
    if self.index >= len(self.names):
        self.quit()
    while(self.confidences[self.index] >= self.THRESHOLD):
        if self.index >= len(self.names):
            self.quit()
        self.index += 1
    img_path = self.names[self.index]
    self.img = ImageTk.PhotoImage(Image.open(img_path))
    self.canvas.create_image(20, 20, anchor=NW, image=self.img)  
    self.canvas.image = self.img
    self.label_text.set(img_path + "\nconfidence: " + str(self.confidences[self.index]))
    self.index += 1

root = Tk()

my_gui = Ethnicity_Labeler(root)
root.mainloop()