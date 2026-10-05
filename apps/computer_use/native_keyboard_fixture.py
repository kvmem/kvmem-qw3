"""Owned native GUI fixture for input qualification; not a business benchmark."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import tkinter as tk


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--nonce',required=True)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(exist_ok=False)
    root=tk.Tk();root.title('QW3 Native Keyboard '+args.nonce)
    root.geometry('760x360+100+120')
    tk.Label(root,text='Native keyboard and paste qualification',font=('sans',20)).pack(pady=15)
    values=[tk.StringVar(),tk.StringVar()]
    entries=[]
    for index,value in enumerate(values):
        tk.Label(root,text='Input '+('A' if index==0 else 'B')).pack()
        entry=tk.Entry(root,textvariable=value,font=('sans',18),width=44,
                       insertontime=0,insertofftime=0)
        entry.pack(pady=10);entries.append(entry)
    tk.Label(root,text='No model requests. Independent fixture state is used only by the qualifier.').pack(pady=20)
    clock=tk.Toplevel(root);clock.overrideredirect(True);clock.geometry('220x70+1000+100')
    label=tk.Label(clock,font=('sans',18),background='#ffe2b0');label.pack(fill='both',expand=True)
    events=[];began=time.monotonic();native_window=None
    process=Path('/proc/self')
    identity=dict(pid=os.getpid(),uid=os.getuid(),start_ticks=int(process.joinpath('stat').read_text().rpartition(')')[2].split()[19]),
                  command=[v.decode() for v in process.joinpath('cmdline').read_bytes().split(b'\0') if v],
                  cwd=os.getcwd(),executable=os.readlink(process/'exe'))

    def publish(*_):
        if native_window is None:return
        selections=[]
        for entry in entries:
            if entry.selection_present():
                first,last=int(entry.index(tk.SEL_FIRST)),int(entry.index(tk.SEL_LAST))
                selections.append(dict(first=first,last=last,text=entry.get()[first:last]))
            else:
                selections.append(None)
        value=dict(nonce=args.nonce,title=root.title(),identity=identity,native_window_id=native_window,
                   entry_window_ids=[e.winfo_id() for e in entries],values=[v.get() for v in values],
                   insert_indices=[int(e.index(tk.INSERT)) for e in entries],
                   selections=selections,
                   key_events=events,published_at=time.time(),pid_property_published_by_fixture=True,
                   clock_is_separate_noninteractive_fixture_window=True)
        temporary=output/'state.tmp';temporary.write_text(json.dumps(value,ensure_ascii=False));temporary.replace(output/'state.json')

    def key(event):
        events.append(dict(keysym=event.keysym,char=event.char,widget_id=event.widget.winfo_id(),at=time.time()))
        root.after_idle(publish)
    for entry in entries:entry.bind('<KeyPress>',key,add='+')
    for value in values:value.trace_add('write',publish)

    def heartbeat():
        label.configure(text='Outside clock\n'+str(int(time.monotonic()-began)))
        publish();root.after(250,heartbeat)

    def ready():
        nonlocal native_window
        root.update_idletasks();root.update()
        # The fixture publishes its actual PID on its own client window.
        # This is declared fixture metadata, not evidence of toolkit support.
        from Xlib import X, Xatom
        from Xlib.display import Display
        from Xlib.protocol.event import ClientMessage
        connection=Display();desktop=connection.screen().root
        clients_atom=connection.intern_atom('_NET_CLIENT_LIST',only_if_exists=True)
        if not clients_atom:
            connection.close();root.after(100,ready);return
        clients=desktop.get_full_property(clients_atom,Xatom.WINDOW)
        if clients is None:
            connection.close();root.after(100,ready);return
        current=connection.create_resource_object('window',root.winfo_id())
        window=None
        for _ in range(16):
            if current.id in clients.value:
                window=current;break
            if current.id==desktop.id:break
            current=current.query_tree().parent
        if window is None:
            connection.close();root.after(100,ready);return
        if window.get_wm_name()!=root.title() or not window.get_wm_class():
            raise RuntimeError('Owned managed client name/class missing')
        window.change_property(connection.intern_atom('_NET_WM_PID'),Xatom.CARDINAL,32,[os.getpid()])
        active_atom=connection.intern_atom('_NET_ACTIVE_WINDOW')
        active=desktop.get_full_property(active_atom,Xatom.WINDOW)
        if active is None or len(active.value)!=1 or int(active.value[0])!=window.id:
            # Request activation through the WM; direct widget focus alone
            # does not establish the WM's active-client property.
            desktop.send_event(ClientMessage(window=window.id,client_type=active_atom,
                data=(32,[2,X.CurrentTime,0,0,0])),event_mask=X.SubstructureRedirectMask|X.SubstructureNotifyMask)
            connection.sync();connection.close();root.after(100,ready);return
        connection.sync();native_window=window.id;connection.close()
        entries[0].focus_set()
        heartbeat()
    root.after(500,ready)
    root.mainloop()


if __name__=='__main__':main()
