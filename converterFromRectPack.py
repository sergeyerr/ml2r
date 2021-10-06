from Base.bp2DState import Action, State
from Base.bp2DBox import Box
from Base.bp2DPnt import Point


def convert(rectPactPacker, bin_size):
    lst = rectPactPacker.rect_list()
    tmp = State(nbins = lst[-1][0] + 1, bin_size = bin_size, boxes_open = [])
    for bin_ind, x, y, b_x, b_y, _ in lst:
        tmp.place_box_in_bin_at_pnt(Box(b_x, b_y), bin_ind, Point(x, y))
    return tmp