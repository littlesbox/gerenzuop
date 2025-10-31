package dome;

import java.util.ArrayList;

public class Database {
	
//	private ArrayList<CD> listcd = new ArrayList<CD>(); 
//	private ArrayList<DVD> listdvd = new ArrayList<DVD>(); 
	private ArrayList<Item> listitem = new ArrayList<Item>();
	
//	public void add(CD cd) {
//		listcd.add(cd);
//	}
//	
//	public void add(DVD dvd) {
//		listdvd.add(dvd);
//	}
//	
//	public void list() {
//		for(CD cd:listcd) {
//			cd.print();
//		}
//		
//		for(DVD dvd:listdvd) {
//			dvd.print();
//		}
//	}
	
	public void add(Item item) {
		listitem.add(item);
	}
	
	public void list() {
		for(Item i:listitem) {
			i.print();
		}
	}

	public static void main(String[] args) {
		Database db = new Database();
		db.add(new CD("abc", "aaa", 4, 30, "123456"));
		db.add(new CD("def", "bbb", 6, 60, "654321"));
		db.add(new DVD("xxx", "mmm", 12, "....."));
		db.add(new DVD("yyy", "nnn", 15, "..."));
		db.add(new VideoGame("game", 100, false, ",,,,", 3));
		db.list();
	}

}