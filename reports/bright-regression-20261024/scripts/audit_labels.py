"""Assistant visual boxes drawn from raw frames; not an independent/blinded review.

These approximate boxes are review evidence, NOT independent human ground truth.
Coordinates refer to the same 720-long-edge image used by inference.
"""
import json
from replay import OUT, save

BOXES = [
    [[546,186,613,401]], [[472,141,544,321]], [[184,203,277,390]], [[428,87,467,182]],
    [[443,136,506,287]], [[425,78,457,160]], [[293,99,339,206]], [[418,110,462,226]],
    [[400,77,438,153]], [[368,141,432,298]], [[397,86,433,156]], [[405,75,438,151]],
    [[405,105,452,218]], [[412,117,457,246]], [[337,130,399,208]], [[326,143,384,308]],
    [[268,154,334,312]], [[339,150,402,313]], [[276,143,350,300]], [[350,160,422,350]],
    [[328,244,417,404]], [[310,165,370,348]], [[405,90,448,188]],
    [[388,69,437,190],[439,12,456,59]], [[579,86,615,188]], [[392,61,440,169]],
    [[289,107,332,214]], [[477,197,542,394]], [[70,207,155,402]],
    [[567,57,605,163],[435,16,459,73],[325,31,355,61]],
    [[492,56,542,172],[356,13,375,62],[324,31,352,60]],
    [[373,210,469,404]], [[490,126,566,296]], [[469,152,590,390]],
    [[362,142,426,215]], [[88,265,215,403]], [[578,104,633,237]],
    [[402,62,446,177],[287,30,315,77]], [[281,95,361,230]], [[286,146,361,298]],
    [[199,106,254,231]],
]
ROBOT_VIDEOS = {1,23,24,25,29,30,36,37,38}


def main() -> None:
    videos = json.loads((OUT / 'protocol.json').read_text())['videos']
    assert len(videos) == len(BOXES) == 41
    labels = []
    for i, (video, boxes) in enumerate(zip(videos, BOXES)):
        labels.append(dict(index=i, video_id=video['id'], source=video['source'], frame=video['audit_frame'],
            person_boxes=boxes, ignore_regions=[[286,10,317,110]] if i == 25 else [],
            ignore_reason='Back person mostly occluded; only head/shoulders visible, full body extent uncertain' if i == 25 else None,
            known_nonperson_boxes=[[48,160,139,384]] if i in ROBOT_VIDEOS else [],
            notes='Foreground humanoid exhibit is not a real person' if i in ROBOT_VIDEOS else ''))
    save(OUT / 'visual-audit-labels.json', dict(labels=labels,
        provenance='Assistant visually inspected 41 raw middle frames and native-resolution far-person crops; approximate boxes, no human GT',
        selection='one middle bright frame from every frozen test video, fixed before replay',
        include_in_training=False, coordinate_system='720-long-edge raw frame pixels',
        matching='one-to-one IoU>=0.5; also report sensitivity at 0.3 and 0.7',
        incomplete_joint_annotation=True, person_instances=sum(map(len,BOXES))))


if __name__ == '__main__':
    main()
