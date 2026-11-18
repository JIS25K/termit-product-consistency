from check import count


def d(box, s=0.6):
    return {"query": "a can", "score": s, "box": box}


def test_duplicates_merge_and_weak_boxes_drop():
    c = count.count_boxes([d((0, 0, 100, 100)), d((5, 5, 102, 98), 0.5), d((300, 0, 400, 100), 0.25)])
    assert c["n"] == 1   # 0.25 < REL * 0.6


def test_three_separate_items():
    assert count.count_boxes([d((0, 0, 100, 100)), d((150, 0, 250, 100)), d((300, 0, 400, 100))])["n"] == 3


def test_group_box_is_not_an_extra_item():
    dets = [d((0, 0, 400, 100), 0.7), d((0, 0, 100, 100)), d((150, 0, 250, 100)), d((300, 0, 400, 100))]
    assert count.count_boxes(dets)["n"] == 3


def test_expected_range_ignores_zero():
    assert count.expected_range([0, 1, 3, 2]) == [1, 3]
    assert count.expected_range([0, 0]) is None
