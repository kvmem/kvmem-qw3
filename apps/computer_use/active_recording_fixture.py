"""Native Tk calibration window; real input changes its color and geometry."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import tkinter as tk


COLORS = ((255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--nonce',required=True)
    args=parser.parse_args()
    directory=args.output.resolve();directory.mkdir(exist_ok=False)
    root=tk.Tk();root.title('QW3 Recording Calibration '+args.nonce)
    canvas=tk.Canvas(root,background='#303030',highlightthickness=0);canvas.pack(fill='both',expand=True)
    phase=0;inputs=[];started=time.monotonic()
    proc=Path('/proc/self');stat=(proc/'stat').read_text().split(') ',1)[1].split()
    identity=dict(pid=os.getpid(),uid=os.getuid(),start_ticks=int(stat[19]),
                  command=sys.argv,cwd=os.getcwd())
    def render():
        width,height=(440,300) if phase%2==0 else (640,440)
        x,y=(150,120) if phase%3==0 else (320,180)
        root.geometry(f'{width}x{height}+{x}+{y}');root.update_idletasks()
        color=COLORS[phase%len(COLORS)]
        canvas.delete('all')
        canvas.create_rectangle(30,60,90,120,fill='#%02x%02x%02x'%color,outline='')
        canvas.create_text(120,90,text=f'Phase {phase}',fill='white',font=('sans',22),anchor='w')
        canvas.create_text(30,150,text='Native keyboard, click and scroll calibration',fill='white',anchor='w')
        canvas.create_text(30,190,text=args.nonce,fill='white',anchor='w')
        canvas.create_text(30,225,text=f'Activity {int(time.monotonic()-started)} seconds',fill='white',anchor='w')
        root.update_idletasks()
        rectangle=[canvas.winfo_rootx()+40,canvas.winfo_rooty()+70,40,40]
        record=dict(nonce=args.nonce,identity=identity,phase=phase,color=list(color),rectangle=rectangle,
                    native_input_events=inputs,title=root.title(),published_at=time.time())
        temporary=directory/'state.tmp';temporary.write_text(json.dumps(record));temporary.replace(directory/'state.json')
    def changed(kind):
        nonlocal phase
        phase+=1;inputs.append(dict(phase=phase,kind=kind,at=time.time()))
        render()
    root.bind('<space>',lambda event:changed('key'))
    canvas.bind('<Button-1>',lambda event:changed('click'))
    canvas.bind('<Button-4>',lambda event:changed('scroll'))
    def heartbeat():
        render();root.after(1000,heartbeat)
    root.after(300,lambda:(root.focus_force(),heartbeat()))
    root.mainloop()


if __name__=='__main__':main()
