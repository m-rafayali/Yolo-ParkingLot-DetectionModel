from parkwatch.geometry import crossed_line, point_in_poly

LINE = [(0, 100), (200, 100)]


def test_crossing_direction():
    assert crossed_line(LINE, (50, 90), (50, 110), "down")
    assert not crossed_line(LINE, (50, 90), (50, 110), "up")
    assert crossed_line(LINE, (50, 110), (50, 90), "any")


def test_no_crossing_outside_segment_or_parallel():
    assert not crossed_line(LINE, (250, 90), (250, 110), "down")  # beyond the end of the line
    assert not crossed_line(LINE, (10, 90), (190, 90), "any")


def test_point_in_poly():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_poly((5, 5), sq)
    assert not point_in_poly((15, 5), sq)
