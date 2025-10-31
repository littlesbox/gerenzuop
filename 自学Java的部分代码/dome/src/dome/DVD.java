package dome;

public class DVD extends Item {
	private String director;

	public DVD(String title, String director, int playlingtime, String comment) {
		super(title, playlingtime, false, comment);
		this.director = director;
	}

	public void print() {
		System.out.print("DVD-- ");
		super.print();
		System.out.print(":");
		System.out.print(director);
		System.out.print("\n");
	}
	
}
