package dome;

public class Item {
	
	private String title;
	private int playlingtime;
	private boolean gotIt = false;
	private String comment;
	
	public Item(String title, int playlingtime, boolean gotIt, String comment) {
		super();
		this.title = title;
		this.playlingtime = playlingtime;
		this.gotIt = gotIt;
		this.comment = comment;
	}

	public Item() {

	}
 
	public void print() {
		System.out.print(title);
	}
	
	public void setTitle(String title) {
		this.title = title;
	}

}
