package dome;

public class CD extends Item {

	private String artist;
	private int numofTracks;
	
	public CD(String title, String artist, int numofTracks, int playlingtime, String comment) {
		super(title, playlingtime, false, comment);
		this.artist = artist;
		this.numofTracks = numofTracks;
	}
	
	public void print() {
		System.out.print("CD-- ");
		super.print();
		System.out.print(":");
		System.out.print(artist);
		System.out.print("\n");
	}
	
	public static void main(String[] args) {
		CD cd = new CD("A", "B", 2, 30, "0000");
		cd.print();
		String a = new String("123");
		String b = new String("123");
//		String a = "123";
//		String b = "123";
		System.out.println(a.equals(b));
		System.out.println(a == b);
	}
	
}
