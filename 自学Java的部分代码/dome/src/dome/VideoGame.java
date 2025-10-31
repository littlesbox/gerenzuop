package dome;

public class VideoGame extends Item {
	
	private int numOfplayers;
	
	public VideoGame(String title, int playlingtime, boolean gotIt, String comment, int number) {
		super(title, playlingtime, gotIt, comment);
		numOfplayers = number;
	}

	@Override
	public void print() {
		System.out.print("VideoGame-- ");
		super.print();
		System.out.print(":");
		System.out.print(numOfplayers);
		System.out.print("\n");
	}

	public static void main(String[] args) {
	
	}

}
