-- Create table
drop table if exists artists;
create table artists (
    mbid STRING,
    artist_mb STRING,
    artist_lastfm STRING,
    country_mb STRING,
    country_lastfm STRING,
    tags_mb STRING,
    tags_lastfm STRING,
    listeners_lastfm BIGINT,
    scrobbles_lastfm BIGINT,
    ambiguous_artist BOOLEAN
)
--     row format serde
--     'org.apache.hadoop.hive.serde2.OpenCSVSerde'
    ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    stored as textfile
    tblproperties (
        'serialization.null.format' = '',
        'skip.header.line.count' = '1');
load data local inpath '/opt/artists.csv' overwrite into table artists;

-- Max scrobble
select t.artist_mb from (select artist_mb, scrobbles_lastfm from artists order by scrobbles_lastfm desc limit 1) t;

-- Most popular tags
drop view  if exists tag_list;
create view tag_list as
    select artist_mb, split(replace(tags_lastfm, ' ', ''), ';') as tag, listeners_lastfm
    from artists;
drop view  if exists tag_popularity;
create view tag_popularity as (
    select t.tag, count(t.tag) as popularity
    from (select explode(tag) as tag from tag_list) t group by t.tag
    );
select * from tag_popularity order by popularity desc limit 10;

-- Most popular artists of most popular tags
drop view if exists top10_tag;
drop view if exists artist_tag;
drop view if exists top_tag_artist;
create view top10_tag as select * from tag_popularity order by popularity desc limit 10;
create view artist_tag as select artist_mb, t as tag, listeners_lastfm from tag_list lateral view explode(tag) adtable as t;
select * from artist_tag limit 10;
create view top_tag_artist as
    select distinct artist_mb, listeners_lastfm
    from artist_tag
    join top10_tag t10t on artist_tag.tag = t10t.tag;
select * from top_tag_artist order by listeners_lastfm desc limit 10;