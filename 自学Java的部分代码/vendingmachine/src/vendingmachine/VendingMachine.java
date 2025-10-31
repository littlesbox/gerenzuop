package vendingmachine;

public class VendingMachine {
	
	int price = 80;
	int balance = f();
	int total;
	
	VendingMachine(){	//构造函数
		total = 1;
	}
	
	VendingMachine(int price){	//重载
		this();	//调用没有参数的构造函数(上面的VendingMachine函数)，只能在第一句，而且只能使用一次
		this.price = price;
	}
	
	int f() {
		return 10;
	}
	
	void setPrice(int price) {
//		price = price;
		this.price = price;
	}
	
	void showPrompt() {
		System.out.println("Welcome");
	}

	void insertMoney(int amount) {
		balance = balance + amount;
		showBalance();
//		this.showBalance();
	}
	
	void showBalance() {
		System.out.println(this.balance);
	}
	
	void getFood() {
		if(balance >= price) {
			System.out.println("Here you are!");
			total = total + price;
			balance = balance - price;
		}
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		VendingMachine vm = new VendingMachine();
		vm.showPrompt();
		vm.showBalance();
		vm.insertMoney(100);
		vm.getFood();
		vm.showBalance();
		
		VendingMachine vm1 = new VendingMachine(100);
		vm1.insertMoney(200);
		vm1.showBalance();
		vm.showBalance();
		
		
	}

}
