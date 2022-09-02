# encoding: utf-8
"""
#@file: ocx.py
#@time: 2022-08-30 19:00
#@author: ywtai 
#@contact: 632910913@qq.com
#@software: PyCharm
#@desc:
"""

import wx
from wx.lib.activexwrapper import MakeActiveXClass
import win32com.client.gencache as win32


class MainFrame(wx.Frame):
    def __init__(self, parent=None, title="test"):
        super().__init__(parent, title=title, size=(900, 800))

        box = wx.Panel(self, -1, style=wx.FULL_REPAINT_ON_RESIZE)
        self._ktocx(box)
        self.Centre()
        self.Show()

    def _ktocx(self, box):
        ocx_classid = '{82192883-43C6-441F-A7C4-4A75B0E8CED5}'
        ktocx = win32.EnsureModule(ocx_classid, 0, 1, 0)
        if ktocx is None:
            wx.MessageBox('未注册', '错误', wx.OK | wx.ICON_WARNING)
            self.Close(True)

        ActiveXWrapper = MakeActiveXClass(ktocx.KTSEDocAxEx)
        self.ocx = ActiveXWrapper(box, -1, size=(945, 810))


if __name__ == "__main__":
    app = wx.App()
    frame = MainFrame()
    app.MainLoop()
