from regressor_set import RegressorSet,DataPoint
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def camelfn(x):
    """
    Code for the Shekel function S5.  The dimensionality
    of x is 2.  The  minimum is at [4.0, 4.0].
    """
    return -(np.sin(x['dim1']*2)+np.abs(x['dim1']-15) + np.sin(x['dim2'])+.2*np.abs(x['dim2']-6))

def optimize_fn():
    config_fname = "camel.yaml"

    with open(config_fname) as yaml_file:
        config = yaml.safe_load(yaml_file)
    print(config)
    regset = RegressorSet(config)
    best_point = None
    all_points = []
    for x in range(25):
        exec_point = regset.generate_next_point()
        out_value = camelfn(exec_point.coord)
        new_point = exec_point.to_data(out_value)
        regset.add_point(new_point)
        print(new_point.fid_level,"\t",new_point.reward, "   \t", list(new_point.coord.values()))
        #print(json.dumps(new_point.to_dict(),indent=2))
        if not best_point or new_point.reward > best_point.reward:
            best_point = new_point
        all_points.append(new_point.coord)
    return all_points

def camelTest():
    """
    Test and visualize DIRECT on a 2D function.  This will draw the contours
    of the target function, the final set of rectangles and mark the optimum
    with a red dot.
    """

    all_points = optimize_fn()

    plt.figure(1)
    plt.clf()
    bounds = [[1.2,28.0], [0.1,13.0]]
    # plot rectangles
    c0 = [(i/50.)*(bounds[0][1]-bounds[0][0])+bounds[0][0] for i in range(51)]
    c1 = [(i/50.)*(bounds[1][1]-bounds[1][0])+bounds[1][0] for i in range(51)]
    z = np.array([[camelfn({"dim1":i,"dim2": j}) for i in c0] for j in c1])

    ax = plt.subplot(111)
    xs = [point['dim1'] for point in all_points]
    ys = [point['dim2'] for point in all_points]
    ax.scatter(xs,ys)

    ax.plot([4.0], [4.0], 'ro')
        # ax.text(rect.center[0], rect.center[1], '%.3f'%rect.y)
    cs = ax.contour(c0, c1, z, 10)
    ax.clabel(cs)
    plt.jet()
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    ax.set_title('final rectangles')

    # ax = plt.subplot(122)
    # fminevol = [y for y,_ in report['fmin evolution']]
    # ax.plot(range(len(fminevol)), fminevol, 'k-', lw=2)
    # ax.set_ylim(min(fminevol)-0.01, max(fminevol)+0.01)
    # ax.grid()
    # ax.set_title('optimization evolution')
    # ax.set_xlabel('iteration')
    # ax.set_ylabel('fmin')
    plt.savefig("fig.png")

if __name__ == "__main__":
    camelTest()
