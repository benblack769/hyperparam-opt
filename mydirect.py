import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import copy

class Rectangle(object):
    """docstring for Rectangle"""
    def __init__(self, lb, ub, y):
        self.lb = copy(lb)
        self.ub = copy(ub)
        self.y = y
        #self.center = (lb+ub)/2#[l+(u-l)/2. for u, l in zip(self.lb, self.ub)]
        self.d = np.sqrt(np.sum(np.square(lb-self.center())))#sum([(l-c)**2. for l, c in zip(self.lb, self.center)])**0.5

    def center(self):
        return (self.lb+self.ub)/2

def direct(f, bounds, maxsample=None):
    def eval_points(xs):
        nonlocal num_samples,best_point
        xprimes = trans(xs)# * (bounds[:,1]-bounds[:,0]) + bounds[:,0]
        ys = f(xprimes)#np.asarray([f(xp) for xp in xprimes])#f(xprimes)
        minidx = np.argmin(ys)
        miny = ys[minidx]
        minx = xs[minidx]

        num_samples += len(xs)

        if len(best_point) == 0 or miny < best_point[0]:
            best_point = [miny,minx]

        return ys

    def trans(x):
        return x * (bounds[:,1]-bounds[:,0]) + bounds[:,0]

    bounds = np.asarray(bounds)


    num_samples = 1
    best_point = []

    N = len(bounds)
    rectangles = []
    first_x = (bounds[:,1]+bounds[:,0])*0.5
    first_sample = eval_points(np.stack([first_x]))[0]
    first = Rectangle(np.zeros(N), np.ones(N), first_sample)
    rectangles.append(first)


    iteration = 0
    epsilon = 10e-10


    while True:
        # print '[direct] iteration %d, samples = %d' % (iteration, SAMPLES)
        if maxsample and num_samples > maxsample:
            print('Reached maximum samples.')
            print(best_point)
            final_point = [best_point[0],trans(best_point[1])]
            report_rects = [Rectangle(trans(r.lb), trans(r.ub), r.y) for r in rectangles]
            return (final_point,{
                "rectangles": report_rects
            })


        # find potentially optimal rectangles
        potopts = []
        for j in range(len(rectangles)):
            maxI1 = None
            minI2 = None
            Rj = rectangles[j]
            for i in  range(len(rectangles)):
                Ri = rectangles[i]
                if i == j: continue
                if Ri.d < Rj.d:
                    # I1
                    val = (Rj.y - Ri.y) / (Rj.d - Ri.d)
                    if maxI1 is None or val > maxI1:
                        maxI1 = val
                elif Ri.d > Rj.d:
                    # I2
                    val = (Ri.y - Rj.y) / (Ri.d - Rj.d)
                    if minI2 is None or val < minI2:
                        minI2 = val
                        if minI2 <= 0.:
                            break
                else:
                    # I3
                    if Rj.y > Ri.y:
                        break

                if maxI1 is not None and minI2 is not None and minI2 < maxI1:
                    break

            else:
                print("argvar")
                # last check...
                if not minI2:
                    potopts.append(Rj)

                elif best_point[0] == 0:
                    if Rj.y <= Rj.d * minI2:
                        potopts.append(Rj)
                elif epsilon <= (best_point[0]-Rj.y)/abs(best_point[0]) + (Rj.d/abs(best_point[0])) * minI2:


                    potopts.append(Rj)

        test_divs = []
        for rect in potopts:
            maxlength = np.max(rect.ub - rect.lb)
            for i in range(N):
                if rect.ub[i] - rect.lb[i] == maxlength:
                    s1 = rect.center()
                    s2 = rect.center()
                    iwidth = rect.ub[i]-rect.lb[i]
                    s1[i] = rect.lb[i] + iwidth / 3.
                    s2[i] = rect.lb[i] + 2. * iwidth / 3.
                    test_divs.append(s1)
                    test_divs.append(s2)

        xs = np.stack(test_divs)
        ys = eval_points(xs)
        print(trans(xs))
        print(ys)

        new_eval_rectangles = []
        new_no_eval_rectangles = []
        idx = 0
        for rect in potopts:
            lengths = rect.ub - rect.lb
            maxlength = np.max(lengths)
            max_count = np.sum(np.equal(lengths,maxlength).astype(np.int32))
            cur_ys = ys[idx: idx+max_count*2]
            cur_xs = xs[idx: idx+max_count*2]
            cur_ys = cur_ys.reshape((max_count,2))
            cur_xs = cur_xs.reshape((max_count,2,len(xs[0])))
            min_ys = np.min(cur_ys,axis=1)
            cidx = 0
            vals = []
            for i in range(N):
                if lengths[i] == maxlength:
                    arg = min_ys[cidx],i#cur_ys[cidx],cur_xs[cidx],i
                    vals.append(arg)
                    cidx += 1
            #vals = [my,y,x,idx for my,y,x,idx,l in zip(min_ys,cur_ys,cur_xs,range(max_count),lengths) if l == max_len]
            vals.sort()

            _,i = vals[0]

            oldrect = rect
            for _,i in vals:
                dwidth = oldrect.ub[i] - oldrect.lb[i]
                split1 = oldrect.lb[i] + dwidth * (1/3)
                split2 = oldrect.lb[i] + dwidth * (2/3)

                idx += 1

                lb1 = copy(oldrect.lb)
                ub1 = copy(oldrect.ub)
                ub1[i] = split1
                new_eval_rectangles.append(Rectangle(lb1, ub1, None))#[l+(u-l)/2. for u, l in zip(lb1, ub1)])))

                lb2 = copy(oldrect.lb)
                ub2 = copy(oldrect.ub)
                lb2[i] = split1
                ub2[i] = split2
                target = Rectangle(lb2, ub2, oldrect.y)

                lb3 = copy(oldrect.lb)
                ub3 = copy(oldrect.ub)
                lb3[i] = split2
                new_eval_rectangles.append(Rectangle(lb3, ub3, None))#samplef([l+(u-l)/2. for u, l in zip(lb3, ub3)])))

                oldrect = target
            new_no_eval_rectangles.append(oldrect)

        new_rect_xs = [rect.center() for rect in new_eval_rectangles]
        new_rect_ys = eval_points(new_rect_xs)
        for y,rect in zip(new_rect_ys,new_eval_rectangles):
            rect.y = y

        for rect in potopts:
            rectangles.remove(rect)

        rectangles += new_eval_rectangles
        rectangles += new_no_eval_rectangles
        print(len(potopts))
        print(len(rectangles))






def demoDIRECT():
    """
    Test and visualize DIRECT on a 2D function.  This will draw the contours
    of the target function, the final set of rectangles and mark the optimum
    with a red dot.
    """

    def foo(x):
        """
        Code for the Shekel function S5.  The dimensionality
        of x is 2.  The  minimum is at [4.0, 4.0].
        """
        return np.sin(x[:,0]*2)+np.abs(x[:,0]-15) + np.sin(x[:,1])+.2*np.abs(x[:,1]-6)

    bounds = [(1.2, 28.), (0.1, 13.)]
    optimum, report = direct(foo, bounds, maxsample=900)

    plt.figure(1)
    plt.clf()

    # plot rectangles
    c0 = [(i/50.)*(bounds[0][1]-bounds[0][0])+bounds[0][0] for i in range(51)]
    c1 = [(i/50.)*(bounds[1][1]-bounds[1][0])+bounds[1][0] for i in range(51)]
    z = np.array([[foo(np.array([[i, j]]))[0] for i in c0] for j in c1])

    ax = plt.subplot(111)
    for rect in report['rectangles']:
        ax.add_artist(plt.Rectangle(rect.lb, rect.ub[0]-rect.lb[0], rect.ub[1]-rect.lb[1], fc='y', ec='k', lw=1, alpha=0.25, fill=True))
        # ax.plot([x[0] for _,x in report['fmin evolution']], [x[1] for _,x in report['fmin evolution']], 'go')
        ax.plot([optimum[1][0]], [optimum[1][1]], 'ro')
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
    plt.show()


if __name__ == "__main__":
    demoDIRECT()
