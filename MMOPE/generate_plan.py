import json

def generate_plan(x_t):
    x = []
    i = 0 
    for x1 in x_t:
        for x2 in x_t:
            x.append([x1, x2])
            i += 1
    p = [ 1 / len(x) ] * len(x)
    return x, p

def write_plan(x, p):
    plan = { "plan": [] }
    for xi, pi in zip(x, p):
        plan["plan"].append({
            "x1": xi[0],
            "x2": xi[1],
            "p": pi
        })
    with open("data2.json", "w") as f:
        f.write(json.dumps(plan))

x_t = [-1, -0.75, -0.25, 0, 0.25, 0.75, 1]
n = len(x_t) ** 2
x, p = generate_plan(x_t)
write_plan(x, p)