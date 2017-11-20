# coding: utf-8

import sys, itertools, json, csv, os
import config
import train

OPTIONS = [
    ("GRADIENT_CLIP", [.01, .1, 1., 10.]),
    ("WEIGHT_DECAY", [.01, .1, 1., 10.]),
    ("MARGIN", [1., 10.]),
]


def main():
    global OPTIONS
    OPTIONS.append(("complete", [0]))
    keys, choices = zip(*OPTIONS)

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        # load data
        plans = load_plans(sys.argv[1])

        for plan in plans:
            if int(plan['complete']) == 0:
                realize_plan(plan)
                plan['complete'] = 1
                save_plans(sys.argv[1], plans, keys)
                train.main(*sys.argv[1:])
                break

    else:
        # create and save new plans
        plans = planning(keys, choices)
        save_plans(sys.argv[1], plans, keys)


def planning(keys, choices):
    plans = [dict(zip(keys, plan)) for plan in itertools.product(*choices)]
    return plans


def save_plans(filename, plans, header):
    with open(filename, 'wb') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(plans)
        f.close()


def load_plans(filename):
    with open(filename) as f:
        reader = csv.DictReader(f)
        plans = [p for p in reader]
        f.close()

    return plans


def realize_plan(plan):
    for k, v in plan.iteritems():
        setattr(config, k, float(v))
        print "set config", k, "with", v


if __name__ == "__main__":
    main()

