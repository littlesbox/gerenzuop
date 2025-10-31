package Fraction;

import java.util.Scanner;

public class Fraction {
	
	int fenzi;
	int fenmu;
	
	Fraction(int fenzi, int fenmu){
		this.fenzi = fenzi;
		this.fenmu = fenmu;
	}
	
	double toDouble() {
		System.out.println(this.fenzi);
		System.out.println(this.fenmu);
		double x = this.fenzi / this.fenmu;
		return x;
	}
	
	Fraction plus(Fraction r) {
		Fraction c = new Fraction(this.fenzi, this.fenmu);
		c.fenzi = c.fenzi + r.fenzi;
		c.fenmu = c.fenmu + r.fenmu;
		return c;
	}
	
	Fraction mutiply(Fraction r) {
		Fraction c = new Fraction(this.fenzi, this.fenmu);
		c.fenzi = c.fenzi * r.fenzi;
		c.fenmu = c.fenmu * r.fenmu;
		return c;
	}
	
	void print() {
		System.out.println(this.fenzi + "/" + this.fenmu);
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner in = new Scanner(System.in);
		Fraction a = new Fraction(in.nextInt(), in.nextInt());
		Fraction b = new Fraction(in.nextInt(), in.nextInt());
		a.print();
		a.plus(b).print();
		a.mutiply(b).print();
		double z = a.toDouble();
		System.out.println(z);
		in.close();
//		System.out.println(1/2);
	}

}
