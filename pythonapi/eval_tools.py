# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import six

from . import anno_tools
from collections import defaultdict


def classification_recall(ground_truth, prediction, recall_n, properties, size_ranges):
    def error(s):
        return {'error': 1, 'msg': s}

    def recall_empty():
        return {'recalls': {n: 0 for n in recall_n}, 'n': 0}

    def recall_add(a, b):
        return {'recalls': {n: a['recalls'][n] + b['recalls'][n] for n in recall_n}, 'n': a['n'] + b['n']}

    stat = dict()
    for szname, _ in size_ranges:
        stat[szname] = {'__all__': recall_empty()}
        for prop in properties:
            stat[szname][prop] = recall_empty()
            stat[szname]['~{}'.format(prop)] = recall_empty()
    chars = defaultdict(recall_empty)
    gts = ground_truth.splitlines()
    prs = prediction.splitlines()
    if len(gts) != len(prs):
        return error('number of lines not match')
    for i, (gt, pr) in enumerate(zip(gts, prs)):
        gt = json.loads(gt)
        try:
            pr = json.loads(pr)
        except:
            return error('line {} is not legal json'.format(i + 1))
        if not isinstance(pr, dict):
            return error('line {} is not json object'.format(i + 1))
        if 'predictions' not in pr:
            return error('line {} does not contain key `predictions`'.format(i + 1))
        gt = gt['ground_truth']
        pr = pr['predictions']
        if not isinstance(pr, list):
            return error('line {} predictions is not an array'.format(i + 1))
        if len(pr) != len(gt):
            return error('line {} wrong predictions length'.format(i + 1))
        for j, (cgt, cpr) in enumerate(zip(gt, pr)):
            if not isinstance(cpr, list):
                return error('line {} prediction {} is not an array'.format(i + 1, j + 1))
            if recall_n[-1] < len(cpr):
                return error('line {} prediction {} contains too few candidates'.format(i + 1, j + 1))
            for k, s in enumerate(cpr):
                if not isinstance(s, six.text_type):
                    return error('line {} prediction {} item {} is not a text'.format(i + 1, j + 1, k + 1))
            thisrc = {'recalls': {n: 1 if cgt['text'] in cpr[:n] else 0 for n in recall_n}, 'n': 1}
            longsize = max(cgt['size'])
            for szname, range in size_ranges:
                if range[0] <= longsize and longsize < range[1]:
                    for prop in properties:
                        if prop not in cgt['properties']:
                            prop = '~{}'.format(prop)
                        stat[szname][prop] = recall_add(stat[szname][prop], thisrc)
                    stat[szname]['__all__'] = recall_add(stat[szname]['__all__'], thisrc)
            chars[cgt['text']] = recall_add(chars[cgt['text']], thisrc)
    return {'error': 0, 'performance': stat, 'group_by_characters': chars}


def iou(bbox_0, bbox_1):  # bbox is represented as (x, y, w, h)
    assert bbox_0[2] >= 0 and bbox_0[3] >= 0 and bbox_1[2] >= 0 and bbox_1[3] >= 0
    A0 = bbox_0[2] * bbox_0[3]
    A1 = bbox_1[2] * bbox_1[3]
    if A0 == 0 or A1 == 0:
        return 0
    Nw = min(bbox_0[0] + bbox_0[2], bbox_1[0] + bbox_1[2]) - max(bbox_0[0], bbox_1[0])
    Nh = min(bbox_0[1] + bbox_0[3], bbox_1[1] + bbox_1[3]) - max(bbox_0[1], bbox_1[1])
    AN = max(0, Nw) * max(0, Nh)
    return AN / (A0 + A1 - AN)


def a_in_b(bbox_0, bbox_1):
    assert bbox_0[2] >= 0 and bbox_0[3] >= 0 and bbox_1[2] >= 0 and bbox_1[3] >= 0
    A0 = bbox_0[2] * bbox_0[3]
    if A0 == 0:
        return 0
    Nw = min(bbox_0[0] + bbox_0[2], bbox_1[0] + bbox_1[2]) - max(bbox_0[0], bbox_1[0])
    Nh = min(bbox_0[1] + bbox_0[3], bbox_1[1] + bbox_1[3]) - max(bbox_0[1], bbox_1[1])
    AN = max(0, Nw) * max(0, Nh)
    return AN / A0


def detection_mAP(ground_truth, detection, properties, size_ranges, max_det, iou_thresh):
    def error(s):
        return {'error': 1, 'msg': s}

    def poly2bbox(poly):
        if 0 == len(poly):
            return [0, 0, 0, 0]
        xmin, ymin = poly[0][0], poly[0][1]
        xmax, ymax = xmin, ymin
        for p in poly:
            xmin, xmax = min(xmin, p[0]), max(xmax, p[0])
            ymin, ymax = min(ymin, p[1]), max(ymax, p[1])
        return [xmin, ymin, xmax - xmin, ymax - ymin]

    def AP_empty():
        return {'n': 0, 'detections': [], 'properties': {prop: {'n': 0, 'recall': 0} for prop in props_all}}

    def AP_compute(m):
        if 0 == m['n']:
            return 0
        acc = []
        rc_inc = []
        m['detections'].sort(key=lambda t: (-t[1], -t[0]))
        match_cnt = 0
        for i, (matched, _) in enumerate(m['detections']):
            assert matched in (0, 1)
            match_cnt += matched
            acc.append(match_cnt / (i + 1))
            rc_inc.append(matched)
        max_acc = 0
        for i in range(len(acc) - 1, -1, -1):
            max_acc = max(max_acc, acc[i])
            acc[i] = max_acc
        AP = 0
        for a, r in zip(acc, rc_inc):
            AP += a * r
        return AP / m['n']

    props_all = properties + ['__all__']
    m = dict()
    for szname, _ in size_ranges:
        m[szname] = defaultdict(AP_empty)

    gts = ground_truth.splitlines()
    dts = detection.splitlines()
    if len(gts) != len(dts):
        return error('number of lines not match')

    for i, (gt, dt) in enumerate(zip(gts, dts)):
        if i % 200 == 0:
            print(i, '/', len(gts))

        gtobj = json.loads(gt)
        try:
            dt = json.loads(dt)
        except:
            return error('line {} is not legal json'.format(i + 1))
        if not isinstance(dt, dict):
            return error('line {} is not json object'.format(i + 1))
        if 'detections' not in dt:
            return error('line {} does not contain key `detections`'.format(i + 1))
        dt = dt['detections']
        if not isinstance(dt, list):
            return error('line {} detections is not an array'.format(i + 1))
        if len(dt) > max_det:
            return error('line {} number of detections exceeds limit ({})'.format(i + 1, max_det))
        for j, char in enumerate(dt):
            if not isinstance(char, dict):
                return error('line {} detection {} is not an object'.format(i + 1, j + 1))
            if 'text' not in char:
                return error('line {} detection {} does not contain key `text`'.format(i + 1, j + 1))
            if 'score' not in char:
                return error('line {} detection {} does not contain key `score`'.format(i + 1, j + 1))
            if 'bbox' not in char:
                return error('line {} detection {} does not contain key `bbox`'.format(i + 1, j + 1))
            if not isinstance(char['text'], six.text_type):
                return error('line {} detection {} text is not text-type'.format(i + 1, j + 1))
            if not isinstance(char['score'], (int, float)):
                return error('line {} detection {} score is neither int nor float'.format(i + 1, j + 1))
            if not isinstance(char['bbox'], list):
                return error('line {} detection {} bbox is not an array'.format(i + 1, j + 1))
            if 4 != len(char['bbox']):
                return error('line {} detection {} bbox is illegal'.format(i + 1, j + 1))
            for t in char['bbox']:
                if not isinstance(t, (int, float)):
                    return error('line {} detection {} bbox is illegal'.format(i + 1, j + 1))
            if char['bbox'][2] <= 0 or char['bbox'][3] <= 0:
                return error('line {} detection {} bbox w or h <= 0'.format(i + 1, j + 1))

        dt.sort(key=lambda o: (-o['score'], o['bbox'], o['text']))
        dt = [(o['bbox'], o['text'], o['score']) for o in dt]

        ig = [(o['bbox'], None) for o in gtobj['ignore']]
        gt = []
        for char in anno_tools.each_char(gtobj):
            if char['is_chinese']:
                gt.append((char['adjusted_bbox'], char['text'], char['properties']))
            else:
                ig.append((poly2bbox(char['polygon']), char['text']))

        matches = []
        dt_ig = [False] * len(dt)
        for j, dtchar in enumerate(dt):
            for k, gtchar in enumerate(gt):
                if dtchar[1] == gtchar[1]:
                    miou = iou(dtchar[0], gtchar[0])
                    if miou > iou_thresh:
                        matches.append((j, k, miou))
                    miou = a_in_b(dtchar[0], gtchar[0])
            for k, igchar in enumerate(ig):
                miou = a_in_b(dtchar[0], igchar[0])
                if miou > iou_thresh:
                    dt_ig[j] = True
        matches.sort(key=lambda t: (-t[2], t[0], t[1]))

        for szname, size_range in size_ranges:
            def in_size(bbox):
                longsize = max(bbox[2], bbox[3])
                return  size_range[0] <= longsize and longsize < size_range[1]

            dt_matched = [0 if in_size(o[0]) and False == b else 2 for o, b in zip(dt, dt_ig)]
            gt_taken = [0 if in_size(o[0]) else 2 for o in gt]
            for i_dt, i_gt, _ in matches:
                if 1 != dt_matched[i_dt] and 1 != gt_taken[i_gt]:
                    if 0 == gt_taken[i_gt]:
                        dt_matched[i_dt] = 1
                        gt_taken[i_gt] = 1
                    else:
                        dt_matched[i_dt] = 2
            for i_dt, (dtchar, match_status) in enumerate(zip(dt, dt_matched)):
                if match_status != 2:
                    m[szname][dtchar[1]]['detections'].append((match_status, dtchar[2]))
            for gtchar, taken in zip(gt, gt_taken):
                if taken != 2:
                    thism = m[szname][gtchar[1]]
                    thism['n'] += 1
                    for prop in gtchar[2]:
                        thism['properties'][prop]['n'] += 1
                        thism['properties'][prop]['recall'] += taken
                    thism['properties']['__all__']['n'] += 1
                    thism['properties']['__all__']['recall'] += taken

    performance = dict()
    for szname, _ in size_ranges:
        n = 0
        mAP = 0
        properties = {prop: {'n': 0, 'recall': 0} for prop in props_all}
        for text, stat in m[szname].items():
            n += stat['n']
            AP = AP_compute(stat)
            mAP += AP * stat['n']
            for prop in props_all:
                properties[prop]['n'] += stat['properties'][prop]['n']
                properties[prop]['recall'] += stat['properties'][prop]['recall']
        assert 0 < n
        performance[szname] = {
            'n': n,
            'mAP': mAP / n,
            'properties': properties,
        }
    return {'error': 0, 'performance': performance}