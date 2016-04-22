function txt = myupdatefcn(empty, event_obj)
  global ru
  txt = {''};
  ru(end+1,:)=event_obj.Position;
  ru=unique(ru,'rows');
  if iseven(size(ru,1))
      x = min(ru(end-1,1),ru(end,1)); y = min(ru(end-1,2),ru(end,2));
      w = abs(ru(end-1,1)-ru(end,1)); h = abs(ru(end-1,2)-ru(end,2)); 
      hold on;
      rectangle('Position',[x y w h],'EdgeColor','r','Tag', 'myRect');   
  elseif (isodd(size(ru,1)) && size(ru,1)>1)
      ru=[];
      delete(findobj(allchild(gcf), 'Tag', 'myRect'))
  end
end