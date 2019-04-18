CREATE TABLE Division(
  name varchar(30),
  Primary Key(name)
);

Create Table Team(
  tid varchar(30),
  name varchar(30),
  divname varchar(30),
  Primary Key(tid),
  Foreign Key(divname) references Division(name) 
);
 
Create Table Player(
  tid varchar(30),
  num int,
  name varchar(30),
  weight int,
  height int,
  dob varchar(30),
  position int,
  Primary Key(tid, num),
  Foreign Key(tid) references Team(tid) 
);

Create Table "Match"(
  matchno int,
  "date" Date,
  guest varchar(30),
  home varchar(30),
  guestscore int,
  homescore int,
  primary key(matchno),
  foreign key(guest) references Team(tid),
  foreign key(home) references Team(tid)
 );
 
 Create Table Played(
   num int,
   tid varchar(30),
   matchno int,
   pts int,
   "field-goal-attempt" int,
   "field-goal-match" int,
   primary key (num, tid, matchno),
   foreign key (tid, num) references Player(tid, num),
   foreign key (matchno) references "Match"(matchno)
  );

Insert into Division values ('Atlantic'), ('Central'), ('Midwest'), ('Pacifik');
Select * from Division;
Update Division set name='Pacific' where name='Pacifik';
Select * from Division;

http://sqlfiddle.com/#!18/7a03e/1