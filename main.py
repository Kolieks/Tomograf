import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import math
import numpy as np
import matplotlib.pyplot as plt
from tkcalendar import DateEntry
import SimpleITK as sitk
import pydicom


def update_coords(angle):
    for counter in range(len(D)):
        D[counter][0] = r * math.cos(angle + math.pi - l / 2 + counter * l / (n - 1)) + S[0]
        D[counter][1] = r * math.sin(angle + math.pi - l / 2 + counter * l / (n - 1)) + S[1]
        E[counter][0] = r * math.cos(angle - l / 2 + counter * l / (n - 1)) + S[0]
        E[counter][1] = r * math.sin(angle - l / 2 + counter * l / (n - 1)) + S[1]

    detectors = np.array(D)
    E.reverse()
    emitter = np.array(E)

    return detectors, emitter


def bresenham_line(start, end, max_iter):
    if max_iter == -1:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1))

    npts, dim = start.shape

    slope = end - start
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    slope_zero = (scale == 0).all(1)
    scale[slope_zero] = np.ones(1)
    normalized_slope = np.array(slope, dtype=np.double) / scale
    normalized_slope[slope_zero] = np.zeros(slope[0].shape)

    step_seq = np.arange(1, max_iter + 1)
    step_mat = np.tile(step_seq, (dim, 1)).T

    line = start[:, np.newaxis, :] + normalized_slope[:, np.newaxis, :] * step_mat

    return np.array(np.rint(line), dtype=start.dtype).reshape(-1, start.shape[-1])


def sinogram(img):
    sin = np.zeros((len(iterations), len(D)))

    plt.figure("Sinogram")
    plt.imshow(sin, cmap='gray')
    plt.axis('off')
    plt.draw()
    plt.pause(0.01)

    for angleIndex, angle in enumerate(iterations):
        detectors, emitters = update_coords(math.radians(angle))

        for emitter in range(len(emitters)):
            dist = np.amax(np.amax(np.abs(emitters[emitter] - detectors[emitter])))

            if np.isnan(dist):
                dist = 0.0

            dist = int(dist)
            pts = bresenham_line(np.array([emitters[emitter]]), np.array([detectors[emitter]]), dist)
            color_value = 0.0

            for point in pts:
                try:
                    pt_0 = int(point[0])
                    pt_1 = int(point[1])
                    color_value += img[pt_0][pt_1]

                except IndexError:
                    pass
            try:
                sin[angleIndex][emitter] = color_value

            except ValueError:
                sin[angleIndex][emitter] = 0.0

        if angleIndex % int((len(iterations) / 30)) == 0 or angleIndex == len(iterations) - 1:
            plt.imshow(sin, cmap='gray')
            plt.draw()
            plt.pause(0.01)

    return sin


def reverse_new_coords(angle):
    for counter in range(len(D)):
        D[counter][0] = r * math.cos(angle + math.pi - l / 2 + counter * l / (n - 1)) + r
        D[counter][1] = r * math.sin(angle + math.pi - l / 2 + counter * l / (n - 1)) + r
        E[counter][0] = r * math.cos(angle - l / 2 + counter * l / (n - 1)) + r
        E[counter][1] = r * math.sin(angle - l / 2 + counter * l / (n - 1)) + r

    detectors = np.array(D)
    E.reverse()
    emitters = np.array(E)

    return detectors, emitters


def output_image(sin):
    square = np.zeros((int(2 * r), int(2 * r)))

    plt.figure('Output image')
    plt.imshow(square, cmap='gray')
    plt.axis('off')
    plt.draw()
    plt.pause(0.01)

    for angle_index, angle in enumerate(iterations):
        detectors, emitters = reverse_new_coords(math.radians(angle))

        for emitter in range(len(emitters)):
            dist = np.amax(np.amax(np.abs(emitters[emitter] - detectors[emitter])))
            if np.isnan(dist):
                dist = 0.0

            dist = int(dist)
            pts = bresenham_line(np.array([emitters[emitter]]), np.array([detectors[emitter]]), dist)
            color_value = sin[angle_index][emitter]

            for point in pts:
                try:
                    pt_0 = int(point[0])
                    pt_1 = int(point[1])
                    square[pt_1][pt_0] = (square[pt_1][pt_0] * emitter + color_value) / (emitter + 1)
                except IndexError:
                    pass
        if angle_index % int((len(iterations) / 30)) == 0 or angle_index == len(iterations) - 1:
            plt.imshow(square, cmap='gray')
            plt.draw()
            plt.pause(0.01)


def upload_file():
    global resized_img
    global img_path
    global window
    global image_label

    img_path = filedialog.askopenfilename(filetypes=[('Jpg Files', '*.jpg'), ('Adobe Fireworks PNG File', '*.png'),
                                                     ("DICOM files", "*.dcm")])
    if len(img_path) > 0:
        if img_path[-1] != 'm':
            img = Image.open(img_path)
            resized_img = img.resize((300, 300))
            resized_img = ImageTk.PhotoImage(resized_img)
            image_label = tk.Label(window, width=300, height=300, image=resized_img)
            image_label.place(x=0, y=0)

            window.geometry('608x680')

        else:
            window.geometry('304x680')
            ds = pydicom.dcmread(img_path)
            try:
                image_label.destroy()
            except:
                pass

            pixel_data = ds.pixel_array
            plt.figure('Sinogram')
            plt.imshow(pixel_data, cmap='gray')
            plt.axis('off')
            dcm_data_window = tk.Toplevel()
            dcm_data_window.title('Dataset')
            dcm_data = tk.Label(master=dcm_data_window, text=str(ds))
            dcm_data.pack()
            plt.show()


def create():
    global alfa, n, l, height, width, r, S, E, D, iterations

    alfa = 180 / int(scans_slider.get())
    n = int(sensors_slider.get())
    l = math.radians(int(range_slider.get()))
    image = plt.imread(img_path)
    height, width, _ = image.shape
    r = width / 2
    S = [width / 2, height / 2]
    E = []
    D = []
    iterations = np.arange(0, 180, alfa)

    for _ in range(n):
        D.append([0, 0])
        E.append([0, 0])

    img = plt.imread(img_path)

    if img.ndim == 3:
        img = img.mean(axis=2)

    img_mirrored = np.flip(img, axis=1)
    img_mirrored = np.rot90(img_mirrored)

    projection = sinogram(img_mirrored)
    output_image(projection)

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    ds = sitk.GetImageFromArray(projection)

    if len(lastname_entry.get()) > 0 and len(name_entry.get()) > 0:
        ds.SetMetaData('0010|0010', f'{lastname_entry.get()}^{name_entry.get()}')
    if len(id_entry.get()) > 0:
        ds.SetMetaData('0010|0020', id_entry.get())
    ds.SetMetaData('0040|A121', date_entry.get_date().strftime('%m/%d/%Y'))
    ds.SetMetaData('0020|000e', "1.2.3.4.5.6.7.8.9")
    ds.SetMetaData('0008|0018', "1.2.3.4.5.6.7.8.9.0")
    ds.SetMetaData('0028|0010', str(projection.shape[0]))
    ds.SetMetaData('0028|0011', str(projection.shape[1]))
    if len(comments_entry.get('1.0', 'end-1c')) > 0:
        comments = comments_entry.get('1.0', 'end-1c').split('\n')
        for idx, comment in enumerate(comments):
            ds.SetMetaData(f'0000|400{idx}', comment)
    ds.SetMetaData('0028|0100', '16')
    ds.SetMetaData('0028|0101', '16')
    ds.SetMetaData('0028|0102', '15')
    ds.SetMetaData('0028|0103', '0')
    ds.SetMetaData('0028|1052', '0')
    ds.SetMetaData('0028|1053', '1')

    writer.SetFileName(f'{date_entry.get_date().strftime("%Y.%m.%d")}-{id_entry.get()}.dcm')
    writer.Execute(ds)


window = tk.Tk()
window.title('Tomograph')

pixel = tk.PhotoImage(width=1, height=1)

frame = tk.Frame(master=window, width=304, height=680)
frame.pack()

scans_slider = Scale(window, from_=1, to=720, length=250, orient=HORIZONTAL)
scans_slider.set(180)
scans_slider.place(x=27, y=400)

sensors_slider = Scale(window, from_=90, to=720, length=250, orient=HORIZONTAL)
sensors_slider.set(180)
sensors_slider.place(x=27, y=480)

range_slider = Scale(window, from_=1, to=180, length=250, orient=HORIZONTAL)
range_slider.set(180)
range_slider.place(x=27, y=560)

scans_text = tk.Label(master=window, text='Scans Number:', image=pixel, width=304, compound='center')
scans_text.place(x=0, y=380)

sensors_text = tk.Label(master=window, text='Sensors Number:', image=pixel, width=304, compound='center')
sensors_text.place(x=0, y=460)

range_text = tk.Label(master=window, text='Range:', image=pixel, width=304, compound='center')
range_text.place(x=0, y=540)

start_y = 150

name_entry = tk.Entry(master=window)
name_entry.place(x=356, y=start_y, width=200)

lastname_entry = tk.Entry(master=window)
lastname_entry.place(x=356, y=start_y + 80, width=200)

id_entry = tk.Entry(master=window)
id_entry.place(x=356, y=start_y + 160, width=200)

date_entry = DateEntry(master=window, width=12, background='#323232', foreground='white', borderwidth=2)
date_entry.place(x=356, y=start_y + 240, width=200)

comments_entry = tk.Text(master=window, height=5)
comments_entry.place(x=356, y=start_y + 320, width=200)

name_text = tk.Label(master=window, text='Patient Name:', image=pixel, width=304, compound='center')
name_text.place(x=304, y=start_y - 25)

lastname_text = tk.Label(master=window, text='Patient Last Name:', image=pixel, width=304, compound='center')
lastname_text.place(x=304, y=start_y + 55)

id_text = tk.Label(master=window, text='Patient ID:', image=pixel, width=304, compound='center')
id_text.place(x=304, y=start_y + 135)

date_text = tk.Label(master=window, text='Examination Date:', image=pixel, width=304, compound='center')
date_text.place(x=304, y=start_y + 215)

comments_text = tk.Label(master=window, text='Comments:', image=pixel, width=304, compound='center')
comments_text.place(x=304, y=start_y + 295)

choose_file_button = tk.Button(text='Choose file', image=pixel, width=100, height=30, compound='center', command=upload_file)
choose_file_button.place(x=102, y=320)

create_button = tk.Button(text='Create', image=pixel, width=100, height=30, compound='center', command=create)
create_button.place(x=102, y=630)

window.resizable(False, False)
window.mainloop()
