# authors: Martin van der Eijk (martin.vandereijk@deltares.nl) en Peter Wellens (p.r.wellens@tudelft.nl)
# version 0.1 230714
# version 0.2 241024

import scipy.integrate as it
import shutil
import scipy.optimize as opt
import scipy.interpolate as intp
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import math
import os
import warnings
import sys
warnings.filterwarnings("ignore")
import time

class cone:
    """
    The initialization of a body with described body interface. A body class should alway contain an initialization
    and a bodyInterface function. Besides the body interface, the width (left B2 and right B1) and the derivative (beta)
    at both sides needs to be defined where separation takes place with respect to the lowest point.
    """
    def __init__(self, Htot, Hmax=0.0, Iyy=0.0, R=0.0, gy=0.0, alpha=0.0, alphaBR=0.0, alphaBL=0.0):
        """
        The initialization of a cone using tangent ogive.
        :param L: The length of the chine
        :param beta: The original deadrise angle of the wedge
        :param alpha: The inclination angle
        """
        self.name = "Bom"
        self.L = R
        self.Htot = Htot
        self.H = Hmax
        self.f = self.bodyInterface
        self.gy0 = gy
        self.I = Iyy
        self.alpha0 = alpha
        self.R = R
        self.B1, self.B2 = self.L, self.L
        self.H1, self.H2 = self.H, self.H
        self.alphaBL = alphaBL
        self.alphaBR = alphaBR
        self.gx, self.gy = -gy * np.sin(alpha * np.pi / 180), gy * np.cos(alpha * np.pi / 180)
        self.interp = self.bodyInterfaceFunc(alpha)

    def bodyInterfaceFunc(self, alpha):
        xlarge = np.linspace(-self.R, self.R, 2000)

        " The function shape "
        rho = (self.R **2 + self.H**2) / (2 * self.R)
        sqrtt = (rho ** 2 - (np.minimum(np.abs(xlarge), self.R) - self.R + rho) ** 2)
        funct = self.H - np.sqrt(sqrtt)

        nnL = 100 * ((90 - alpha) < alphaBL)
        nnR = 100 * ((90 + alpha) < alphaBR)
        xlarge = np.hstack([-np.linspace(self.R, self.R, nnL), xlarge, np.linspace(self.R, self.R, nnR)])
        funct = np.hstack([np.linspace(self.Htot, self.H + (self.Htot - self.H) / nnL, nnL), funct,
                           np.linspace(self.H + (self.Htot - self.H) / nnR, self.Htot, nnR)])

        " Rotation"
        origin = self.rotation(np.array([xlarge, funct]), alpha)

        " Position lowest point "
        yindx = np.where(origin[1, :] == np.min(origin[1, :]))[0][0]
        origin_old = origin.copy()
        origin[1, :] -= origin[1, yindx]
        origin[0, :] -= origin[0, yindx]
        self.gx, self.gy = -self.gy0 * np.sin(alpha * np.pi / 180) - origin_old[0, yindx], self.gy0 * np.cos(alpha * np.pi / 180) - origin_old[1, yindx]

        " Check "
        dfdx = self.dinterface(origin[1, :], origin[0, :])
        dfRdx = np.tan((self.alphaBR) * np.pi / 180)
        dfLdx = -np.tan((self.alphaBL) * np.pi / 180)
        xmin, xmax = np.where(origin[0, :] == np.min(origin[0, :]))[0][0], np.where(np.abs(dfdx - dfRdx) == np.min(np.abs(dfdx - dfRdx)))[0][0]
        if np.sum(dfdx >= dfLdx) != 0.0:
            xmin = np.where(dfdx >= dfLdx)[0][0]
        if np.sum(dfdx >= dfRdx) != 0.0:
            xmax = np.where(dfdx >= dfRdx)[0][0]
        self.B1 = origin[0, xmax]
        self.H1 = origin[1, xmax]
        self.B2 = abs(origin[0, xmin])
        self.H2 = origin[1, xmin]

        " Redo to prevent strange extrapolation "
        xlarge = xlarge[xmin:xmax+1]
        rho = (self.R **2 + self.H**2) / (2 * self.R)
        sqrtt = (rho ** 2 - (np.minimum(np.abs(xlarge), self.R) - self.R + rho) ** 2)
        funct = self.H - np.sqrt(sqrtt)
        funct[xmin:nnL+xmin] = np.linspace(self.Htot, self.H + (self.Htot - self.H) / (nnL), nnL)
        funct[(xmax-xmin)+1-nnR:(xmax-xmin)+1] = np.linspace(self.Htot, self.H + (self.Htot - self.H) / (nnR), nnR)
        origin = self.rotation(np.array([xlarge, funct]), alpha)

        " Position lowest point "
        yindx = np.where(origin[1, :] == np.min(origin[1, :]))[0][0]
        origin[1, :] -= origin[1, yindx]
        origin[0, :] -= origin[0, yindx]

        "Interpolation"
        return intp.interp1d(origin[0, :], origin[1,:], fill_value='extrapolate')
    def bodyInterface(self, x):
        """
        The body interface describing a cone using tangent ogive with x-output.
        :param x: the x position which results in output the y-position with the lowest point (0, 0).
        :return:
        """
        # origin = np.interp(x, origin[0, :], origin[1, :])
        return self.interp(x)

    def dinterface(self, f, x):
        """
        Derivative of the function interface over x.
        :param x: the x-coordinate gives the vertical position of the interface
        :param h: tolerance
        :return:
        """
        return (f[1:] - f[:-1]) / (x[1:] - x[:-1])

    def rotation(self, dat, alpha):
        angle = np.pi/180 * alpha
        xx = dat[0, :] * np.cos(angle) - dat[1, :] * np.sin(angle)
        yy = (dat[0, :] * np.sin(angle) + dat[1, :] * np.cos(angle))
        return np.array([xx, yy])

class wettedArea:
    """
    In this class the wetted area and its derivative based on height is calculated for a given body.
    The fictituous body is added based on the angle imposed.
    A choice needs to be made what kind of fictituous body you want.
    The body defined is extended with the fictitous one.
    """
    def __init__(self, body, alpha1, alpha2, interface="Linear"):
        """
        Initializatino of the wetted area.

        :param body: initialization from the class "body" with bodyInterface
        :param alpha1: rightside angle of separation.
        :param alpha2: leftside angle of separation.
        :param interface: the choice of interface for the fictitous body. (linear or higher)
        """
        self.body = body
        self.bodyInterface = body.f
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        if interface == "Linear" or interface == "linear":
            self.fluidInterface = self.linfluidInterface
        else:
            print("Wrong input interface fluid model")
            exit()

        " Plot the lines"
        gg_body = []
        gg_fluid2 = []
        gg_fluid1 = []

        " Assumption made that the fictitous continuation remains horizontal "
        x = np.linspace(-2*self.body.B2, self.body.B1 * 2, 2000)
        B1, B2 = self.body.B1, self.body.B2
        for xx in x:
            if -B2 <= xx <= B1:
                gg_body.append([xx, self.interface(body.alpha0, xx)])
            elif xx < -B2:
                gg_fluid1.append([xx, self.interface(body.alpha0, xx)])
            elif xx > B1:
                gg_fluid2.append([xx, self.interface(body.alpha0, xx)])

        gg_body = np.array(gg_body)
        gg_fluid1 = np.array(gg_fluid1)
        gg_fluid2 = np.array(gg_fluid2)

        plt.figure()
        plt.plot(gg_body[:, 0], gg_body[:, 1], 'r',
                 gg_fluid1[:, 0], gg_fluid1[:, 1], 'b',
                 self.body.gx, self.body.gy, 'g*',
                 gg_fluid2[:, 0], gg_fluid2[:, 1], 'b')
        plt.xlim(-2*B2, 2*B1)
        plt.ylim(-0.01*body.H, body.Htot*2)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.legend(["Body", "Fluid", "CoR"])
        plt.xlabel("X position [m] ")
        plt.ylabel("Y position [m] ")
        plt.savefig(nameOutput +'/Initial_interface.png')

    def linfluidInterface(self, alpha, x):
        """
        The linear fluid interface for fictitous body based on one angle as input from the initialization.
        :param x: the x-coordinate gives the vertical position of the interface extended with bodyinterface itself.
        :return:
        """
        B1, B2 = self.body.B1, self.body.B2
        H1, H2 = self.body.H1, self.body.H2

        return (x < 0.0) * ((-x - B2) * np.tan((self.alpha2) * np.pi / 180) + H2) + \
               (x > 0.0) * ((x - B1) * np.tan((self.alpha1) * np.pi / 180) + H1)

    def interface(self, alpha, x):
        """
        Combines the body interface and fluid interface based on the separation points defined in class body.
        :param x: the x-coordinate gives the vertical position of the interface
        :return:
        """
        infaceBody = self.bodyInterface(x)
        B1, B2 = self.body.B1, self.body.B2
        inface = (x > B1) * self.fluidInterface(alpha, x) + \
                 (x < -B2) * self.fluidInterface(alpha, x) + \
                 (-B2 <= x) * (x <= B1) * infaceBody
        return inface

    def dinterface(self, x, alpha, h=1e-5):
        """
        Derivative of the function interface over x.
        :param x: the x-coordinate gives the vertical position of the interface
        :param h: tolerance
        :return:
        """
        return (self.interface(alpha, x + h) - self.interface(alpha, x - h)) / (2 * h)

    def finterface1(self, x, p, alpha):
        """
        Function 1 multiplied by term to solve integral from -1 to 1 for wetted length.
        :param x: the x-coordinate gives the vertical position of the interface
        :param p: the vector containing the wetted lengths [c1, c2]
        :return:
        """
        c1, c2 = p
        gg = 0.5 * (c1 + c2) * x + 0.5 * (c1 - c2)
        term = np.divide(1-x, 1+x, out=np.zeros_like(x), where=1 + x != 0.0)
        return (self.interface(alpha, gg)) * np.sqrt(term)

    def finterface2(self, x, p, alpha):
        """
        Function 2 multiplied by term to solve integral from -1 to 1 for wetted length.
        :param x: the x-coordinate gives the vertical position of the interface
        :param p: the vector containing the wetted lengths [c1, c2]
        :return:
        """
        c1, c2 = p
        gg = 0.5 * (c1 + c2) * x + 0.5 * (c1 - c2)
        term = np.divide(1+x, 1-x, out=np.zeros_like(x), where=1 - x != 0.0)
        return (self.interface(alpha, gg)) * np.sqrt(term)

    def dfinterface1(self, x, p, alpha):
        """
        Function 1 derivative multiplied by term to solve integral from -1 to 1 for wetted length derivative.
        :param x: the x-coordinate gives the vertical position of the interface
        :param p: the vector containing the wetted lengths [dc1dh, dc2dh]
        :return:
        """
        c1, c2 = p
        gg = 0.5 * (c1 + c2) * x + 0.5 * (c1 - c2)
        term = np.divide(1+x, 1-x, out=np.zeros_like(x), where=1 - x != 0.0)
        return (self.dinterface(gg, alpha)) * np.sqrt(term) * (0.5 * x + 0.5)

    def dfinterface2(self, x, p, alpha):
        """
        Function 2 derivative multiplied by term to solve integral from -1 to 1 for wetted length derivative.
        :param x: the x-coordinate gives the vertical position of the interface
        :param p: the vector containing the wetted lengths [dc1dh, dc2dh]
        :return:
        """
        c1, c2 = p
        gg = 0.5 * (c1 + c2) * x + 0.5 * (c1 - c2)
        term = np.divide(1-x, 1+x, out=np.zeros_like(x), where=1 + x != 0.0)
        return (self.dinterface(gg, alpha)) * np.sqrt(term) * (0.5 * x - 0.5)

    def fintegrate(self, fun, x2, x1, p, alpha):
        """
        Integrating function from x2 to x1 depending on the wetted length or derivative p
        :param fun: the function to integrate depending on p
        :param x2: low part interval
        :param x1: high part interval
        :param p: the vector containing the wetted lengths
        :return:
        """
        return it.quad(fun, x2, x1, args=(p, alpha))[0]

    def integrate(self, p, zz, alpha):
        """
        Integration of the body + fictituous body splitted in parts for solving the wetted length.
        Includes dependency on waterdepth.
        :param p: the vector containing the wetted lengths
        :param zz: water depth wrt still water surface.
        :return:
        """

        B1, B2 = self.body.B1, self.body.B2
        term1 = np.minimum((2 * B1 - (p[0] - p[1])) / (p[0] + p[1]), 1)
        term2 = np.minimum(-(-2 * B2 - (p[0] - p[1])) / (p[0] + p[1]), 1)
        return [self.fintegrate(self.finterface1, -term2, term1, p, alpha) +
                self.fintegrate(self.finterface1, -1, -term2, p, alpha) +
                self.fintegrate(self.finterface1, term1, 1, p, alpha) - np.pi * zz,
                self.fintegrate(self.finterface2, -term2, term1, p, alpha) +
                self.fintegrate(self.finterface2, -1, -term2, p, alpha) +
                self.fintegrate(self.finterface2, term1, 1, p, alpha) - np.pi * zz]
    def dintegrate(self, p, alpha):
        """
        Integration of the body + fictituous body splitted in parts for solving the derivative of wetted length.
        :param p: the vector containing the wetted lengths derivatives
        :return:
        """

        B1, B2 = self.body.B1, self.body.B2
        term1 = np.minimum((2 * B1 - (p[0] - p[1])) / (p[0] + p[1]), 1)
        term2 = np.minimum(-(-2 * B2 - (p[0] - p[1])) / (p[0] + p[1]), 1)
        return [self.fintegrate(self.dfinterface1, -term2, term1, p, alpha) +
                self.fintegrate(self.dfinterface1, -1, -term2, p, alpha) +
                self.fintegrate(self.dfinterface1, term1, 1, p, alpha),
                self.fintegrate(self.dfinterface2, -term2, term1, p, alpha) +
                self.fintegrate(self.dfinterface2, -1, -term2, p, alpha) +
                self.fintegrate(self.dfinterface2, term1, 1, p, alpha)]
    def initialWettedlength(self, x, alpha, zz=0.0):
        """
        Function defining the wetted length based on an undisturbed free surface without jets.
        :param x: the x-coordinate gives the vertical position of the interface
        :param zz: water depth wrt still water surface.
        :return:
        """
        return self.interface(alpha, x) - zz

    def rootWettedLength(self, zz, alpha, c10=0.01, c20=-0.01):
        """
        Root finder for the wetted length. Including initialization based on without jets as input for root finder.
        :param zz: water depth wrt still water surface.
        :param c10: initial value for right side wetted length
        :param c20: initial value for left side wetted length
        :return:
        """
        c10, c20 = opt.fsolve(self.initialWettedlength, [c10, c20], args=(alpha, zz))
        c10, c20 = abs(c10), abs(c20)
        " Determine root "
        self.body.interp = self.body.bodyInterfaceFunc(alpha)
        c1, c2 = opt.root(self.integrate, [c10, c20], args=(zz, alpha)).x

        return c1, c2, c10, c20

    def derWettedLength(self, p, alpha):
        """
        Derivative of wetted length calculation with input the wetted length.
        :param p: the vector containing the wetted lengths
        :return:
        """
        term = np.array(self.dintegrate(p, alpha))
        return np.divide(np.pi, term, out=np.zeros_like(term), where=term != 0.0)

class pressureEstimate:
    """
    Class to determine the pressure distribution for every component over time and space.
    """
    def __init__(self, wet, density, spacestep):
        self.body = wet.body
        self.wet = wet
        self.density = density
        self.spacestep = spacestep
        np.seterr(all="ignore")

    def valuesWet(self, h, alpha):
        c1, c2, _, _ = self.wet.rootWettedLength(h, alpha)
        dc1dh, dc2dh = self.wet.derWettedLength([c1, c2], alpha)


        B1, B2 = self.body.B1, self.body.B2
        x = np.linspace(-np.minimum(c2, B2), np.minimum(c1, B1), 2*self.spacestep)
        return [c1, c2], [dc1dh, dc2dh], x

    def pressureTot(self, h, V, A, alpha):
        c, dcdh, x = self.valuesWet(h, alpha)
        Pv = np.maximum(self.pressureSlam(V, np.array([c]), np.array([dcdh]), x) + self.pressureJet(V, np.array([c]), x, alpha), 0)
        Pa = self.pressureAm(A, np.array([c]), x, alpha)
        return Pv + Pa, self.pressureSlam(V, np.array([c]), np.array([dcdh]), x), self.pressureJet(V, np.array([c]), x, alpha), Pa

    def pressureAm(self, A, c, x, alpha):
        c1, c2 = c[:, 0], c[:, 1]


        B1, B2 = self.body.B1, self.body.B2
        term1 = np.minimum(B1, c1)
        term2 = np.minimum(B2, c2)
        term = np.divide(x + term2, term1 + term2, out=np.zeros_like(x + term2), where=abs(term1 + term2) > 1e-10)
        ff = self.wet.interface(alpha, -term2) + term * \
             (self.wet.interface(alpha, term1) - self.wet.interface(alpha, -term2))
        bodyF = self.wet.interface(alpha, x) - ff
        return self.density * A * (np.sqrt(np.maximum((np.minimum(c1, B1) - x) *
                                                (np.minimum(c2, B2) + x), 0)) +
                                        bodyF)

    def pressureSlam(self, V, c, dcdh, x, eps=1e-10):
        c1, c2 = c[:, 0], c[:, 1]
        dc1dh, dc2dh = dcdh[:, 0], dcdh[:, 1]
        term1 = np.divide(c2 + x, c1 - x, out=np.zeros_like(c2 + x), where=abs(c1 - x) > eps)
        term2 = np.divide(c1 - x, c2 + x, out=np.zeros_like(c1 - x), where=abs(c2 + x) > eps)
        value = 0.5 * self.density * V**2 * (dc1dh * np.sqrt(term1) + dc2dh * np.sqrt(term2))
        value[np.isnan(value)] = 0.0
        return value

    def pressureJet(self, V, c, x, alpha, eps=1e-10):
        c1, c2 = c[:, 0], c[:, 1]
        fx = self.wet.dinterface(x, alpha)
        dphidx = - V * np.divide((c1 - c2 - 2 * x), 2 * np.sqrt((c2 + x)*(c1 - x)), out=np.zeros_like(0.5 *(c1 - c2 - 2 * x)), where=abs((c2 + x)*(c1 - x)) > eps) - V * fx
        dphidt = V **2
        return - self.density * (dphidt + dphidx * V * fx / (1 + fx ** 2) \
                                 + 1 / (2 * (1 + fx ** 2)) * (dphidx ** 2 - V ** 2))

    def pressureHydro(self, h, x, alpha):
        term = np.maximum(h - self.wet.interface(alpha, x), 0)
        return self.density * 9.81 * term * 0.56 # Op basis van literatuur eigenonderzoek (Eijk Wellens 2023)

class forceEstimate:
    """
    Class to calculate all force components using the pressure distributions integrated.
    """
    def __init__(self, pres):
        self.pres = pres
        self.body = pres.body
    def valuesWet(self, h, alpha):
        c1, c2, _, _ = self.pres.wet.rootWettedLength(h, alpha)
        dc1dh, dc2dh = self.pres.wet.derWettedLength([c1, c2], alpha)
        return [c1, c2], [dc1dh, dc2dh]

    def Fam(self, A, c, alpha):
        c1, c2 = c[:, 0], c[:, 1]


        B1, B2 = self.body.B1, self.body.B2
        x = np.linspace(-np.minimum(c2, B2), np.minimum(c1,B1), 2*self.pres.spacestep)
        ff = np.trapz(self.pres.wet.interface(alpha, x), x=x)
        # term1= (np.minimum(c1, B1))
        # term2= (np.minimum(c2, B2))
        # return self.pres.density * A * (ff + (term1 + term2)**2 * np.pi/8 - h * (term1 + term2))
        return np.trapz(self.pres.pressureAm(A, c, x, alpha), x=x, axis=0)

    def Fhyd(self, h, c, alpha):
        c1, c2 = c[:, 0], c[:, 1]

        B1, B2 = self.body.B1, self.body.B2
        x = np.linspace(-np.minimum(c2, B2), np.minimum(c1, B1), 2 * self.pres.spacestep)
        return np.trapz(self.pres.pressureHydro(h, x, alpha), x=x, axis=0)

    def Fslam(self, V, c, dcdh, alpha):
        c1, c2 = c[:, 0], c[:, 1]

        B1, B2 = self.body.B1, self.body.B2
        x = np.linspace(-np.minimum(c2, B2), np.minimum(c1, B1), 2*self.pres.spacestep)
        return np.trapz(self.pres.pressureSlam(V, c, dcdh, x), x=x, axis=0)

    def Fjet(self, V, c, alpha):
        c1, c2 = c[:, 0], c[:, 1]

        B1, B2 = self.body.B1, self.body.B2
        x = np.linspace(-np.minimum(c2, B2), np.minimum(c1, B1), 2*self.pres.spacestep)
        return np.trapz(self.pres.pressureJet(V, c, x, alpha), x=x, axis=0)

    def Mom(self, h, V, A, c, dcdh, alpha):
        c1, c2 = c[:, 0], c[:, 1]

        B1, B2 = self.body.B1, self.body.B2

        x = np.linspace(-np.minimum(c2, B2), np.minimum(c1, B1), 2*self.pres.spacestep)
        Momx = np.trapz((self.pres.pressureAm(A, c, x, alpha) + self.pres.pressureHydro(h, x, alpha) + np.maximum(self.pres.pressureJet(V, c, x, alpha) + self.pres.pressureSlam(V, c, dcdh, x), 0)) * (x-self.body.gx), x=x-self.body.gx, axis=0)
        x1 = np.linspace(-np.minimum(c2, B2), 0, self.pres.spacestep)
        x2 = np.linspace(0, np.minimum(c1, B1), self.pres.spacestep)
        Momy1 = np.trapz(self.pres.wet.dinterface(x1, alpha) * (self.pres.pressureAm(A, c, x1, alpha) + self.pres.pressureHydro(h, x1, alpha) + np.maximum(self.pres.pressureJet(V, c, x1, alpha) + self.pres.pressureSlam(V, c, dcdh, x1), 0)) * (self.pres.wet.interface(alpha, x1)-self.body.gy), x=x1, axis=0)
        Momy2 = np.trapz(self.pres.wet.dinterface(x2, alpha) * (self.pres.pressureAm(A, c, x2, alpha) + self.pres.pressureHydro(h, x2, alpha) + np.maximum(
            self.pres.pressureJet(V, c, x2, alpha) + self.pres.pressureSlam(V, c, dcdh, x2), 0)) * (
                                     self.pres.wet.interface(alpha, x2) - self.body.gy), x=x2, axis=0)
        return Momx+Momy1+Momy2

    def Fhor(self, h, V, A, c, dcdh, alpha):
        """
        The horizontal force for assymetrical shapes by multiplying the derivative of the body interface over x
        with the pressure and integrate over the area. Is trustworthy for deadrise angles below 30 degrees.
        :param h: depth position of the keel w.r.t. still free surface.
        :param V: speed of body
        :param A: acceleration of body
        :param c: wetted length [right, left] of the body depending on time.
        :param dcdh: derivative wetted length over depth [right, left] of the body depending on time.
        :return:
        """
        c1, c2 = c[:, 0], c[:, 1]

        B1, B2 = self.body.B1, self.body.B2
        x = np.linspace(-np.minimum(c2, B2), np.minimum(c1, B1), 2*self.pres.spacestep)
        return -np.trapz(self.pres.wet.dinterface(x, alpha) * (self.pres.pressureAm(A, c, x, alpha) + self.pres.pressureHydro(h, x, alpha) + np.maximum(self.pres.pressureJet(V, c, x, alpha) + self.pres.pressureSlam(V, c, dcdh, x), 0)), x=x, axis=0)

    def Ftot(self, h, V, A, c, dcdh, alpha):
        c1, c2 = c[:, 0], c[:, 1]

        B1, B2 = self.body.B1, self.body.B2
        x = np.linspace(-np.minimum(c2, B2), np.minimum(c1, B1), 2*self.pres.spacestep)
        return np.trapz(self.pres.pressureAm(A, c, x, alpha) + self.pres.pressureHydro(h, x, alpha) + np.maximum(self.pres.pressureJet(V, c, x, alpha) + self.pres.pressureSlam(V, c, dcdh, x), 0), x=x, axis=0)

    def forceTot(self, h, V, A, alpha):
        c, dcdh = self.valuesWet(h, alpha)
        return self.Ftot(h, V, A, np.array([c]), np.array([dcdh]), alpha), self.Fhyd(h, np.array([c]), alpha), self.Fslam(V, np.array([c]), np.array([dcdh]), alpha), self.Fjet(V, np.array([c]), alpha), self.Fam(A, np.array([c]), alpha)

    def fhorTot(self, h, V, A, alpha):
        c, dcdh = self.valuesWet(h, alpha)
        return self.Fhor(h, V, A, np.array([c]), np.array([dcdh]), alpha)

    def momTot(self, h, V, A, alpha):
        c, dcdh = self.valuesWet(h, alpha)
        return self.Mom(h, V, A, np.array([c]), np.array([dcdh]), alpha)


class EOM:
    def __init__(self, body, wet, force, time, hmax, mass=1e16, eps=1e-8):
        self.mass = mass if mass != 0 else 1e16
        self.eps = eps
        self.body = body
        self.wet = wet
        self.force = force
        self.hmax = hmax
        self.t = None
        self.EOM = self.loopEOM
        self.t = time
    def heighVelAcc(self, h0, dhdt0, d2hd2t, dt, alpha0, alpha10, alpha2):
        dhdt = dhdt0 + d2hd2t * dt
        h = h0 + dhdt * dt

        alpha1 = alpha10 + alpha2 * dt
        alpha0 = alpha0 * np.pi / 180
        alpha = alpha0 + alpha1 * dt
        alpha = alpha * 180 / np.pi

        c1, c2, _, _= self.wet.rootWettedLength(h, alpha)
        dc1dh, dc2dh = self.wet.derWettedLength([c1, c2], alpha)
        return h, dhdt, alpha, alpha1, [c1, c2], [dc1dh, dc2dh]

    def heighVelAccx(self, h0, dhdt0, d2hd2t, dt):
        dhdt = dhdt0 + d2hd2t * dt
        h = h0 + dhdt * dt

        return h, dhdt

    def coreEOM(self, h, dhdt, d2hd2t, x, dxdt, d2xd2t, c, dcdh, n, alpha, dalphadt):
        Ftot = self.force.Ftot(h, dhdt, d2hd2t, np.array([c]), np.array([dcdh]), alpha) * 2. * self.body.R * np.pi / 4.
        Mom = self.force.Mom(h, dhdt, d2hd2t, np.array([c]), np.array([dcdh]), alpha) * 2. * self.body.R * np.pi / 4.
        Fx = self.force.Fhor(h, dhdt, d2hd2t, np.array([c]), np.array([dcdh]), alpha)
        Fa = self.force.Fam(d2hd2t, np.array([c]), alpha)
        Fs = self.force.Fslam(dhdt, np.array([c]), np.array([dcdh]), alpha)
        Fj = self.force.Fjet(dhdt, np.array([c]), alpha)
        Fh = self.force.Fhyd(h, np.array([c]), alpha)
        Ptot = self.force.pres.pressureAm(d2hd2t, np.array([c]), x, alpha) + \
               np.maximum(self.force.pres.pressureJet(dhdt, np.array([c]), x, alpha) +
                          self.force.pres.pressureSlam(dhdt, np.array([c]), np.array([dcdh]), x), 0)
        d2hd2t_new = -Ftot[0] / self.mass + 9.81
        d2alphad2t_new = Mom[0] / self.body.I
        d2xd2t_new = Fx[0] / self.mass
        d2xd2t_new = 0.0 # There is no restoring force!

        if n < len(self.t) - 1:
            dt = self.t[1 + n] - self.t[0 + n]
        else:
            dt = self.t[-1] - self.t[-2]
        h, dhdt, alpha, dalphadt, c, dcdh = self.heighVelAcc(h, dhdt, d2hd2t_new, dt, alpha, dalphadt, d2alphad2t_new)
        x, dxdt = self.heighVelAccx(x, dxdt, d2xd2t_new, dt)
        return dt, h, dhdt, d2hd2t_new, x, dxdt, d2xd2t_new, c, dcdh, Ftot, Fa, Fh, Fj, Fs, Fx, Mom, Ptot, alpha, dalphadt

    def loopEOM(self, dhdt, dxdt=0.0, d2hd2t=0.0, d2xd2t=0.0, alpha=0.0):
        B1, B2 = self.body.B1, self.body.B2
        print(" ")
        print("EOM voor het volgende object:")
        print(self.body.name, " wordt gesimuleerd")
        print(" ")

        print("Simulatie.....")
        dt = self.t[1] - self.t[0]
        tt = dt
        h = dhdt * dt
        x = dxdt * dt
        c1, c2, _, _= self.wet.rootWettedLength(h, alpha)
        dcdh = self.wet.derWettedLength([c1, c2], alpha)
        dalphadt = 0.0

        t_list = []
        h_list, dhdt_list, d2hd2t_list = [], [], []
        x_list, dxdt_list, d2xd2t_list = [], [], []
        F_list, Fa_list, Fs_list, Fj_list = [], [], [], []
        Fx_list, Mmom_list = [], []
        Fh_list = []
        Ptot_list, alpha_list = [], []
        c_list, dcdh_list = [], []
        Drag_list = []
        n = 0

        while self.hmax > h and alpha<90.0:
            n = n + 1
            dt, h_new, dhdt, d2hd2t, x, dxdt, d2xd2t, c, dcdh, Ftot, Fa, Fh, Fj, Fs, Fx, Mmom, Ptot, alpha, dalphadt = self.coreEOM(h=h, dhdt=dhdt, d2hd2t=d2hd2t,
                                                                                           x=x, dxdt=dxdt, d2xd2t=d2xd2t,
                                                                                          c=[c1, c2], dcdh=dcdh, n=n, alpha=alpha, dalphadt=dalphadt)
            Drag = np.sqrt(Ftot ** 2 + Fx ** 2) / \
                   (0.5 * 1000 * np.round(np.sqrt(dhdt ** 2 + dxdt ** 2), 5) ** 2 * self.body.R ** 2 * np.pi)

            "OUTPUT SAVE"
            alpha_list.append(alpha)
            F_list.append(Ftot)
            Fa_list.append(Fa)
            Fh_list.append(Fh)
            Fj_list.append(Fj)
            Fs_list.append(Fs)
            Fx_list.append(Fx)
            Mmom_list.append(Mmom)
            h_list.append(h_new)
            dhdt_list.append(dhdt)
            d2hd2t_list.append(d2hd2t)
            x_list.append(x)
            dxdt_list.append(dxdt)
            d2xd2t_list.append(d2xd2t)
            c_list.append(c)
            dcdh_list.append(dcdh)
            t_list.append(tt)
            Ptot_list.append(Ptot)
            Drag_list.append(Drag[0])
            tt += dt
            h = h_new
            c1, c2 = c
            print(' ')
            print("Hoek: ", 90 - np.round(alpha, 0), "[grad]")
            print("Snelheid: ", np.round(np.sqrt(dhdt**2 + dxdt**2), 5), "[m/s]")
            print("Tijd:", np.round(tt, 5), "[s]")
            print("Diepgang:", np.round(h / self.hmax * 100, 1), "[%]")

        self.t, self.h, self.V, self.A, self.xh, self.xV, self.xA, self.c, self.dcdh, self.F, self.Fa, self.Fh, self.Fs, self.Fj, self.Fx, self.Mom, self.Ptot, self.alpha, self.Drag = \
            np.array(t_list), np.array(h_list), np.array(dhdt_list), np.array(d2hd2t_list), \
            np.array(x_list), np.array(dxdt_list), np.array(d2xd2t_list), np.array(c_list),\
            np.array(dcdh_list), np.array(F_list), np.array(Fa_list), np.array(Fh_list), np.array(Fs_list), np.array(Fj_list), \
            np.array(Fx_list), np.array(Mmom_list), np.array(Ptot_list), np.array(alpha_list), np.array(Drag_list)

        print("Finished and stored in dataset")
        print("---------")
        print(" ")

class airFall:
    def __init__(self, body, mass, density_air, gravity, Z0, V0):
        self.Az = body.R ** 2 * np.pi   #PRW 241024 corrected from v0.1: body.R ** 2 * np.pi * 0.25
        self.dens_air = density_air
        self.grav = gravity
        self.mass = mass
        self.Cd_air = 0.23              #PRW 241024 corrected from v0.1: 0.18 (now in accordance with achtergrond/241024-peter_wellens-afwerpmunitie_eindsnelheid_ten_behoeve_van_weerstandscoefficient.xlsx)
        self.Z0 = Z0
        self.V0 = -V0

    def dvdt(self, t, v):
        return [v[1], - self.grav + 1 / 2 * self.Cd_air * self.Az * self.dens_air / self.mass * v[1] ** 2]

def reach_depth(t, v): return v[0]


class input:
    def __init__(self):
        " Data read "
        mass = np.array([100, 250, 500, 1000]) # Mass in lb
        L = np.array([0.737, 0.701, 0.945, 1.334]) # Length of projectile
        D = np.array([0.208, 0.259, 0.328, 0.41]) # Diameter of projectile
        Lmax_per = np.array([0.26, 0.44, 0.44, 0.311]) # Length percentage to reach maximum diameter

        #install openpyxl and pandas
        " Reading files "
        df = pd.read_excel(os.getcwd()+'/input.xlsx')
        data = pd.DataFrame(df, columns=['Invoerparameter', 'Waarde','Eenheid','Range'])
        self.Sit = data.loc[(data['Invoerparameter'] == 'Type bombardement/berekening')]['Waarde'][1]
        if self.Sit not in ['Duik', 'Tapijt']:
            print("WAARSCHUWING: geen waarde voor type bombardement/berekening. Berekening is gestopt.")
            exit()

        " Sizing bomb and water "
        self.mass = data.loc[(data['Invoerparameter'] == 'Kaliber')]['Waarde'][0]
        if math.isnan(self.mass):
            print("WAARSCHUWING: geen waarde voor kaliber. Berekening is gestopt.")
            exit()
        idx = np.where(mass == self.mass)[0]
        self.L = L[idx][0]
        self.D = D[idx][0]
        self.Lmax = Lmax_per[idx][0]
        self.alpha0 = 90 - data.loc[(data['Invoerparameter'] == 'Inslaghoek t.o.v. horizontaal')]['Waarde'][3]
        self.dpt = data.loc[(data['Invoerparameter'] == 'Waterdiepte')]['Waarde'][4]
        self.H0 = data.loc[(data['Invoerparameter'] == 'Afwerphoogte')]['Waarde'][2] * 0.3048 # in meters
        self.V0 = 0.0
        self.xV0 = 0.0

        " Determine situation "
        if self.Sit == 'Duik':
            self.V0 = 201.39 * (np.cos(self.alpha0 * np.pi / 180))
            if math.isnan(self.H0):
                self.H0 = 0.3048 * 3000
        elif self.Sit == 'Tapijt' and math.isnan(self.H0):
            print("WAARSCHUWING: geen waarde ingevuld voor afwerphoogte. Berekening is gestopt.")
            exit()
        if math.isnan(self.dpt) or math.isnan(self.alpha0):
            print("WAARSCHUWING: geen waarde voor waterdiepte of invalshoek. Berekening is gestopt.")
            exit()

if __name__ == '__main__':
    start = time.time()

    " Input "
    inp = input()
    nameOutput = repr(inp.Sit)+"_"+repr(inp.mass) +"lb_"+repr(np.round(inp.H0, 0))+"m_"+repr(inp.dpt)+"m_"+repr(90-inp.alpha0)+"grad"
    print(nameOutput)

    " Generate folders and copy input "
    if not os.path.exists(nameOutput):
        os.makedirs(nameOutput)
        os.makedirs(nameOutput+"/Results")

    shutil.copy("input.xlsx", nameOutput)
    shutil.copy(os.path.basename(sys.argv[0]), nameOutput)

    " Data for simulation "
    alpha0 = float(inp.alpha0) # Inclination angle
    alphaBR = 70.65  # FBC right side, rotates with alpha0! Not larger than 89
    alphaBL = 70.65  # FBC left side, rotates with alpha0! Not larger than 89
    V0 = inp.V0 # Impact speed
    xV0 = inp.xV0
    R = inp.D / 2 # Radius cone
    H = inp.L # Height cone
    Hmax = inp.Lmax * inp.L # Height cone maximum radius flow sep
    mass3D = 0.45359 * float(inp.mass) # Mass 3D cone
    depth = float(inp.dpt) # Maximum depth

    " Rest of data calculation, approximation "
    mass = mass3D
    gy = 0.55 * H # Height CoR assuming 55 percent of height cone
    A0 = 0.0
    Iyy = 1/12 * mass3D * (H ** 2 + 3 * R ** 2)

    " Definition of classes "
    cone1 = cone(alpha=alpha0, Htot=inp.L, Hmax=Hmax, gy=gy, Iyy=Iyy, R=R, alphaBR=alphaBR, alphaBL=alphaBL)
    air1 = airFall(body=cone1, mass=mass3D, density_air=1.2, gravity=9.81, Z0=inp.H0, V0=V0)
    wet1 = wettedArea(body=cone1, alpha1=alphaBR, alpha2=alphaBL, interface='linear')
    pres1 = pressureEstimate(wet=wet1, density=1000, spacestep=1000)
    for1 = forceEstimate(pres=pres1)

    " EOM "
    " eom air "
    reach_depth.terminal = True
    sol = it.solve_ivp(air1.dvdt, t_span=(0, 1e6), y0=[air1.Z0, air1.V0], events=reach_depth, max_step=0.1)
    V0 = abs(sol.y[1, -2] + (sol.y[1, -1] - sol.y[1, -2]) / (sol.y[0, -1] - sol.y[0, -2]) * (0.0 - sol.y[0, -2]))

    np.savetxt(nameOutput + "/Results/SnelheidLucht.csv", np.array(["Snelheidlucht", V0]).T, fmt="%s",delimiter=',')
    print(" ")
    print("RESULTAAT:")
    print("Snelheid bij aanraking wateroppervlak: ", V0,'[m/s]')

    " eom water "
    aa = 0.001
    steps_small = 1e-4
    steps_big = 1e-4
    ttt = np.hstack([np.linspace(0.0, aa, int(np.round(aa / steps_small) + 1)),
                     np.linspace(aa + steps_big, 20000, int(np.round((20000 - aa) / steps_big)))])
    # Timeline such that it stops after reaching maximum depth
    eom1 = EOM(body=cone1, wet=wet1, force=for1, hmax=depth, time=ttt, mass=mass, eps=1e-5)
    eom1.EOM(dhdt=V0, dxdt=xV0, d2hd2t=A0, alpha=alpha0)

    " FINISH "
    end = time.time()
    print("Simulatie tijd: ", np.round(end - start, 1),"[s]")

    " Output csv "
    Hoek = 90 - np.round(eom1.alpha[-2] + (eom1.alpha[-1] - eom1.alpha[-2]) / (eom1.h[-1] - eom1.h[-2])*(inp.dpt - eom1.h[-2]), 0)
    V1, V2 = np.sqrt(eom1.V[-1]**2 + eom1.xV[-1]**2), np.sqrt(eom1.V[-2]**2 + eom1.xV[-2]**2)
    Snelheid = np.round(V1 + (V1 - V2) / (eom1.h[-1] - eom1.h[-2])*(inp.dpt - eom1.h[-2]), 1)

    print(" ")
    print("RESULTAAT:")
    print("De resultaten voor: "+repr(inp.Sit)+"_"+repr(inp.mass) +"lb_"+repr(np.round(inp.H0, 0))+"m_"+repr(inp.dpt)+"m_"+repr(90-inp.alpha0)+"grad")
    if eom1.alpha[-1]>=90.0:
        print("De munitie ligt op de waterbodem.")
        np.savetxt(nameOutput + "/Results/SnelheidWater_Hoek.csv",
                   np.array([["Snelheid [m/s]", 0], ["Hoek [grad]", 0]]).T, fmt="%s", delimiter=',')
    else:
        print("Snelheid en hoek bij aanraking waterbodem: ", Snelheid,'[m/s]', Hoek, '[grad]')
        np.savetxt(nameOutput + "/Results/SnelheidWater_Hoek.csv", np.array([["Snelheid [m/s]", Snelheid],["Hoek [grad]", Hoek]]).T, fmt="%s", delimiter=',')

    # " Output csv "
    np.savetxt(nameOutput + "/Results/Vel.csv", np.array([eom1.t, eom1.V]).T, delimiter=',')
    np.savetxt(nameOutput + "/Results/VelHor.csv", np.array([eom1.t, eom1.xV]).T, delimiter=',')
    np.savetxt(nameOutput + "/Results/Height.csv", np.array([eom1.t, -eom1.h]).T, delimiter=',')
    np.savetxt(nameOutput + "/Results/HeightHor.csv", np.array([eom1.t, -eom1.xh]).T, delimiter=',')
    np.savetxt(nameOutput + "/Results/IncAngle.csv", np.array([eom1.t, eom1.alpha]).T, delimiter=',')
    np.savetxt(nameOutput + "/Results/Force.csv", np.array([eom1.t, eom1.F[:, 0]]).T, delimiter=',')
    np.savetxt(nameOutput + "/Results/Moment.csv", np.array([eom1.t, eom1.Mom[:, 0]]).T, delimiter=',')
    np.savetxt(nameOutput + "/Results/Acc.csv", np.array([eom1.t, eom1.A]).T, delimiter=',')
    np.savetxt(nameOutput + "/Results/ForceHor.csv", np.array([eom1.t, eom1.Fx[:, 0]]).T, delimiter=',')
    np.savetxt(nameOutput + "/Results/AccHor.csv", np.array([eom1.t, eom1.xA]).T, delimiter=',')
    np.savetxt(nameOutput + "/Results/Drag.csv", np.array([eom1.t, eom1.Drag]).T, delimiter=',')
