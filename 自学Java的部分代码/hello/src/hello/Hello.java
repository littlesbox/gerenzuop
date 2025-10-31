package hello;

import java.util.Scanner;

public class Hello {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.out.println("你好");
		Scanner in = new Scanner(System.in);
//		System.out.println("echo:" + in.nextLine());
		
		int x = 0;
		int[] numbers = new int[100];
//		System.out.println(price);
		
		double sum = 0;
		int cnt = 0;
		x = in.nextInt();
		while(x != -1) {
			numbers[cnt] = x;
			sum += x;
			cnt++;
			x=in.nextInt();
		}
		if(cnt > 0) {
			double average = sum / cnt;
			for (int i = 0 ; i<cnt; i++) {
				if(numbers[i]>average) {
					System.out.println(numbers [i]);
				}
			}
			System.out.println(sum/cnt);
		}
		
	}

}
